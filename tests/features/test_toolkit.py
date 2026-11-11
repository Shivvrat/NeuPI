"""Tests for the query helpers, evaluation, saving/loading solvers and training callbacks."""

import itertools
import json

import numpy as np
import pytest
import torch
from neupi import (
    MLP,
    DiscreteEmbedder,
    KNearestDiscretizer,
    ProbabilisticGraphicalModel,
    QueryDataset,
    SelfSupervisedTrainer,
    SinglePassInferenceEngine,
    SumProductNetwork,
    ThresholdDiscretizer,
    compare_assignments,
    exact_inference,
    load_solver,
    make_query_masks,
    mpe_log_likelihood_loss,
    save_solver,
    score_assignments,
)
from neupi.core.model import BaseNNModel
from test_robustness import SPN_DIR, random_model, reference_log_likelihood, reference_spn, write_uai
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------------------
# Query masks and datasets
# ----------------------------------------------------------------------------------------


def assert_partition(evidence, query, unobs):
    total = evidence.int() + query.int() + unobs.int()
    assert (total == 1).all()


def test_masks_from_ratios_are_a_shared_partition():
    evidence, query, unobs = make_query_masks(5, 10, query_ratio=0.3, unobs_ratio=0.2, seed=0)
    assert evidence.shape == query.shape == unobs.shape == (5, 10)
    assert evidence.dtype == torch.bool
    assert_partition(evidence, query, unobs)
    assert (query.sum(1) == 3).all() and (unobs.sum(1) == 2).all()
    # One split shared by all queries, reproducible from the seed
    assert (query == query[0]).all()
    assert torch.equal(query, make_query_masks(5, 10, query_ratio=0.3, unobs_ratio=0.2, seed=0)[1])


def test_masks_per_example_differ_between_queries():
    evidence, query, unobs = make_query_masks(50, 20, query_ratio=0.5, per_example=True, seed=1)
    assert_partition(evidence, query, unobs)
    assert (query.sum(1) == 10).all()
    assert not (query == query[0]).all()
    assert not unobs.any()


def test_masks_from_indices():
    evidence, query, unobs = make_query_masks(3, 6, query_vars=[0, 2], unobs_vars=[5])
    assert query[:, [0, 2]].all() and query.sum() == 6
    assert unobs[:, 5].all() and unobs.sum() == 3
    assert evidence[:, [1, 3, 4]].all() and evidence.sum() == 9
    # Fixed query variables with randomly chosen unobserved ones from the rest
    evidence, query, unobs = make_query_masks(4, 6, query_vars=[0], unobs_ratio=0.5, per_example=True)
    assert_partition(evidence, query, unobs)
    assert query[:, 0].all() and not unobs[:, 0].any() and (unobs.sum(1) == 3).all()


def test_a_small_positive_ratio_gives_at_least_one_variable():
    _, query, _ = make_query_masks(2, 10, query_ratio=0.01)
    assert (query.sum(1) == 1).all()
    _, query, _ = make_query_masks(2, 10, query_ratio=0.0)
    assert not query.any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"query_ratio": 0.5, "query_vars": [0]},
        {"query_ratio": 1.5},
        {"query_ratio": 0.7, "unobs_ratio": 0.5},
        {"query_vars": [10]},
        {"query_vars": [0, 1], "unobs_vars": [1]},
        {"query_ratio": 0.5, "unobs_ratio": 0.1, "unobs_vars": [0]},
    ],
)
def test_invalid_mask_arguments_raise(kwargs):
    with pytest.raises(ValueError):
        make_query_masks(2, 10, **kwargs)


def test_query_dataset_hides_the_non_evidence_values():
    samples = torch.ones(4, 6, dtype=torch.int64)
    dataset = QueryDataset.from_split(samples, query_ratio=0.5, unobs_ratio=1 / 6, seed=0)
    assert len(dataset) == 4
    evidence_data, evidence_mask, query_mask, unobs_mask = dataset[0]
    assert evidence_data.dtype == torch.float32 and evidence_mask.dtype == torch.bool
    assert (evidence_data[evidence_mask] == 1).all()
    assert (evidence_data[~evidence_mask] == 0).all()
    batch = next(iter(DataLoader(dataset, batch_size=3)))
    assert [t.shape for t in batch] == [(3, 6)] * 4


def test_query_dataset_accepts_shared_masks_and_checks_inputs():
    samples = torch.randint(0, 2, (5, 4))
    evidence = torch.tensor([True, True, False, False])
    dataset = QueryDataset(samples, evidence, ~evidence)
    assert dataset.query_mask.shape == (5, 4) and not dataset.unobs_mask.any()
    with pytest.raises(ValueError, match="binary"):
        QueryDataset(samples * 2 + 1, evidence, ~evidence)
    with pytest.raises(ValueError, match="overlap"):
        QueryDataset(samples, evidence, evidence)
    with pytest.raises(ValueError, match="shape"):
        QueryDataset(samples, torch.ones(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))


# ----------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------


@pytest.fixture
def small_pgm(tmp_path):
    rng = np.random.default_rng(0)
    num_vars = 7
    scopes, tables = random_model(rng, num_vars, [1, 1, 2, 2, 2, 3, 3])
    pgm = ProbabilisticGraphicalModel(write_uai(tmp_path / "m.uai", num_vars, scopes, tables), device=DEVICE)
    return pgm, num_vars, scopes, tables


def test_exact_inference_matches_enumeration(small_pgm):
    pgm, num_vars, scopes, tables = small_pgm
    samples = torch.randint(0, 2, (6, num_vars), generator=torch.Generator().manual_seed(0))
    dataset = QueryDataset.from_split(samples, query_ratio=0.6, per_example=True, seed=2)
    evidence_data, evidence_mask, query_mask, unobs_mask = dataset[:]
    assignments, scores = exact_inference(pgm, evidence_data, evidence_mask, query_mask)

    for i in range(len(dataset)):
        q = query_mask[i].nonzero().flatten().tolist()
        candidates = np.tile(evidence_data[i].numpy().astype(np.int64), (2 ** len(q), 1))
        candidates[:, q] = list(itertools.product([0, 1], repeat=len(q)))
        best = reference_log_likelihood(candidates, scopes, tables).max()
        assert scores[i].item() == pytest.approx(best, rel=1e-4, abs=1e-4)
        # The returned assignment keeps the evidence and has the returned score
        assert torch.equal(assignments[i][evidence_mask[i]], evidence_data[i][evidence_mask[i]])
        assert reference_log_likelihood(assignments[i : i + 1].long().numpy(), scopes, tables)[0] == pytest.approx(
            best, rel=1e-4, abs=1e-4
        )
    assert torch.allclose(score_assignments(pgm, assignments), scores)


def test_exact_inference_marginalizes_unobserved_variables():
    path = SPN_DIR / "nltcs/spn.json"
    data = json.loads(path.read_text())
    spn = SumProductNetwork(str(path), device=DEVICE)
    samples = torch.randint(0, 2, (3, spn.num_var), generator=torch.Generator().manual_seed(0))
    dataset = QueryDataset.from_split(samples, query_ratio=0.25, unobs_ratio=0.25, per_example=True, seed=0)
    evidence_data, evidence_mask, query_mask, unobs_mask = dataset[:]
    _, scores = exact_inference(spn, evidence_data, evidence_mask, query_mask, unobs_mask)

    for i in range(len(dataset)):
        q = query_mask[i].nonzero().flatten().tolist()
        candidates = np.tile(evidence_data[i].numpy(), (2 ** len(q), 1))
        candidates[:, q] = list(itertools.product([0, 1], repeat=len(q)))
        candidates[:, unobs_mask[i].numpy()] = np.nan
        assert scores[i].item() == pytest.approx(reference_spn(data, candidates).max(), rel=1e-4, abs=1e-4)


def test_exact_inference_refuses_large_queries(small_pgm):
    pgm, num_vars, _, _ = small_pgm
    evidence, query, unobs = make_query_masks(1, num_vars, query_ratio=1.0)
    with pytest.raises(ValueError, match="max_query_vars"):
        exact_inference(pgm, torch.zeros(1, num_vars), evidence, query, unobs, max_query_vars=3)


def test_compare_assignments_reports_gaps_to_the_reference(small_pgm):
    pgm, num_vars, _, _ = small_pgm
    samples = torch.randint(0, 2, (8, num_vars), generator=torch.Generator().manual_seed(1))
    evidence_data, evidence_mask, query_mask, _ = QueryDataset.from_split(samples, query_ratio=0.5, seed=0)[:]
    exact, exact_scores = exact_inference(pgm, evidence_data, evidence_mask, query_mask)
    threshold = ThresholdDiscretizer()(torch.full((8, num_vars), 0.7), query_mask)
    threshold = torch.where(evidence_mask, evidence_data, threshold)

    results = compare_assignments(
        pgm,
        {"exact": exact, "threshold": threshold},
        reference="exact",
        evidence_data=evidence_data,
        evidence_mask=evidence_mask,
    )
    assert results["exact"]["mean_gap"] == pytest.approx(0.0, abs=1e-6)
    assert results["threshold"]["mean_gap"] >= -1e-5
    assert results["threshold"]["mean_log_likelihood"] <= results["exact"]["mean_log_likelihood"] + 1e-5
    # A tensor of reference scores gives the same gaps
    by_tensor = compare_assignments(pgm, {"threshold": threshold}, reference=exact_scores)
    assert by_tensor["threshold"]["mean_gap"] == pytest.approx(results["threshold"]["mean_gap"], abs=1e-5)
    with pytest.raises(ValueError, match="evidence"):
        compare_assignments(pgm, {"bad": 1 - exact}, evidence_data=evidence_data, evidence_mask=evidence_mask)


# ----------------------------------------------------------------------------------------
# Saving and loading solvers, training callbacks
# ----------------------------------------------------------------------------------------


def train_solver(pgm, num_vars, epochs=2, **fit_kwargs):
    samples = torch.randint(0, 2, (32, num_vars), generator=torch.Generator().manual_seed(0))
    dataset = QueryDataset.from_split(samples, query_ratio=0.5, seed=0)
    embedding = DiscreteEmbedder(num_vars, query_embedding=0.5)
    model = MLP([16, 8], num_vars, embedding, use_batchnorm=True, dropout_rate=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = SelfSupervisedTrainer(model, pgm, mpe_log_likelihood_loss, optimizer, device=DEVICE)
    trainer.fit(DataLoader(dataset, batch_size=8), num_epochs=epochs, **fit_kwargs)
    return trainer, dataset


def test_saved_solver_loads_with_the_same_outputs(small_pgm, tmp_path):
    pgm, num_vars, _, _ = small_pgm
    trainer, dataset = train_solver(pgm, num_vars, verbose=False)
    model = trainer.model.eval()
    path = tmp_path / "solver.pt"
    save_solver(path, model, metadata={"model_file": "m.uai", "query_ratio": 0.5})

    loaded, metadata = load_solver(path, device=DEVICE, return_metadata=True)
    assert metadata == {"model_file": "m.uai", "query_ratio": 0.5}
    assert not loaded.training
    assert loaded.embedding.query_embedding == 0.5
    batch = [t.to(DEVICE) for t in dataset[:]]
    with torch.no_grad():
        assert torch.allclose(model(*batch), loaded(*batch))
    engine_out = SinglePassInferenceEngine(loaded, KNearestDiscretizer(pgm, k=2), device=DEVICE).run(dataset[:])
    assert engine_out["final_assignments"].shape == (len(dataset), num_vars)


def test_saving_rejects_unregistered_models_and_other_files(tmp_path):
    class Custom(BaseNNModel):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(2, 2)

        def forward(self, x):
            return self.layer(x)

    with pytest.raises(ValueError, match="registered"):
        save_solver(tmp_path / "c.pt", Custom())
    torch.save({"weights": torch.zeros(2)}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="save_solver"):
        load_solver(tmp_path / "other.pt")


def test_fit_verbose_callback_and_history(small_pgm, capsys):
    pgm, num_vars, _, _ = small_pgm
    seen = []

    def stop_after_two(info):
        seen.append(info)
        return info["epoch"] == 2

    trainer, _ = train_solver(pgm, num_vars, epochs=5, verbose=False, callback=stop_after_two)
    assert capsys.readouterr().out == ""
    assert [info["epoch"] for info in seen] == [1, 2]
    assert trainer.history == seen
    assert set(seen[0]) == {"epoch", "num_epochs", "train_loss", "val_loss", "lr", "seconds"}
    assert seen[0]["val_loss"] is None and seen[0]["num_epochs"] == 5

    train_solver(pgm, num_vars, epochs=1)
    assert "Epoch 1/1" in capsys.readouterr().out
