"""Tests for the configurable k, OAUAI oracle, entropy loss, validation selection and ITSELF options."""

import functools
from pathlib import Path

import numpy as np

import pytest
import torch
from neupi import (
    MLP,
    BestOfDiscretizer,
    SumProductNetwork,
    OAUAI,
    DiscreteEmbedder,
    ITSELF_Engine,
    KNearestDiscretizer,
    MarkovNetwork,
    SelfSupervisedTrainer,
    ThresholdDiscretizer,
    entropy_penalty,
    mpe_log_likelihood_loss,
)
from neupi.core.discretizer import resolve_k, validate_k
from neupi.utils.pgm_utils import apply_evidence, evaluate_log_likelihood
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def mn_evaluator():
    uai_path = Path(__file__).parent.parent.parent / "examples/networks/mn/Grids_17.uai"
    return MarkovNetwork(uai_file=str(uai_path), device=DEVICE)


def make_batch(num_vars, num_samples=4, seed=0):
    gen = torch.Generator().manual_seed(seed)
    evidence_data = torch.randint(0, 2, (num_samples, num_vars), generator=gen).float()
    evidence_mask = torch.rand(num_samples, num_vars, generator=gen) > 0.5
    query_mask = ~evidence_mask
    unobs_mask = torch.zeros_like(evidence_mask)
    return [t.to(DEVICE) for t in (evidence_data, evidence_mask, query_mask, unobs_mask)]


def make_model(num_vars, use_batchnorm=True):
    torch.manual_seed(0)
    return MLP(
        hidden_sizes=[16],
        output_size=num_vars,
        embedding=DiscreteEmbedder(num_vars),
        use_batchnorm=use_batchnorm,
    ).to(DEVICE)


# --- k ---


def test_resolve_k():
    assert resolve_k(5, 100) == 5
    assert resolve_k(0.1, 100) == 10
    assert resolve_k(0.1, 95) == 10  # ceil
    assert resolve_k(0.1, 3) == 1  # at least 1
    assert resolve_k(0.1, 0) == 0
    for bad in (0, -1, 0.0, 1.0, 1.5, True, "3"):
        with pytest.raises((TypeError, ValueError)):
            validate_k(bad)
    # numpy integers are counts
    assert validate_k(np.int64(1024)) == 1024 and resolve_k(np.int64(5), 100) == 5
    # max_k caps the per-query k: SINE's min(|Q| / 4, 200)
    assert resolve_k(0.25, 400, max_k=200) == 100
    assert resolve_k(0.25, 1300, max_k=200) == 200
    assert resolve_k(300, 1300, max_k=200) == 200
    for bad_max in (0, 1.5, True):
        with pytest.raises(ValueError):
            validate_k(0.25, bad_max)


def test_mlp_has_no_batchnorm_by_default(mn_evaluator):
    n = mn_evaluator.num_variables
    model = MLP(hidden_sizes=[8], output_size=n, embedding=DiscreteEmbedder(n))
    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules())


def test_entropy_aggregate(mn_evaluator):
    n = mn_evaluator.num_variables
    batch = make_batch(n, num_samples=4)
    trainers = {}
    for agg in ("avg", "sum"):
        model = make_model(n)
        trainers[agg] = SelfSupervisedTrainer(
            model=model,
            pgm_evaluator=mn_evaluator,
            loss_fn=functools.partial(mpe_log_likelihood_loss, aggregate=agg),
            optimizer=torch.optim.Adam(model.parameters()),
            device=DEVICE,
            entropy_weight=1.0,
            entropy_aggregate=agg,
        )
        trainers[agg].model.eval()
    with torch.no_grad():
        avg = trainers["avg"]._compute_loss(batch)
        total = trainers["sum"]._compute_loss(batch)
    assert torch.allclose(total, 4 * avg, rtol=1e-5)


def test_oauai_fractional_k_changes_only_query_vars(mn_evaluator):
    n = mn_evaluator.num_variables
    probs = torch.rand(3, n, device=DEVICE)
    query_mask = torch.zeros(3, n, dtype=torch.bool, device=DEVICE)
    query_mask[0, :20] = True
    query_mask[1, :40] = True
    query_mask[2, :2] = True
    other = torch.zeros_like(query_mask)

    thresholded = (probs >= 0.5).float()
    out = OAUAI(pgm_evaluator=mn_evaluator, k=0.1)(probs, query_mask, other, other)
    assert torch.equal(out[~query_mask], thresholded[~query_mask])
    assert torch.all(mn_evaluator(out) >= mn_evaluator(thresholded))


def test_oauai_int_k_capped_at_query_vars(mn_evaluator):
    n = mn_evaluator.num_variables
    probs = torch.rand(2, n, device=DEVICE)
    query_mask = torch.zeros(2, n, dtype=torch.bool, device=DEVICE)
    query_mask[:, :2] = True
    other = torch.zeros_like(query_mask)
    out = OAUAI(pgm_evaluator=mn_evaluator, k=5)(probs, query_mask, other, other)
    assert torch.equal(out[:, 2:], (probs[:, 2:] >= 0.5).float())


def test_oauai_oracle(mn_evaluator):
    n = mn_evaluator.num_variables
    probs = torch.rand(2, n, device=DEVICE)
    query_mask = torch.zeros(2, n, dtype=torch.bool, device=DEVICE)
    query_mask[:, :30] = True
    other = torch.zeros_like(query_mask)
    calls = []

    def enumerating_oracle(assignment, free_indices):
        # An exact oracle over the free variables, by brute force
        calls.append(free_indices.clone())
        k = len(free_indices)
        grid = torch.tensor(
            [[(j >> b) & 1 for b in range(k)] for j in range(2**k)],
            dtype=assignment.dtype,
            device=assignment.device,
        )
        candidates = assignment.unsqueeze(0).repeat(2**k, 1)
        candidates[:, free_indices] = grid
        return candidates[mn_evaluator(candidates).argmax()]

    with_oracle = OAUAI(pgm_evaluator=mn_evaluator, k=4, oracle=enumerating_oracle)
    out = with_oracle(probs, query_mask, other, other)
    assert len(calls) == 2 and all(len(c) == 4 and torch.all(c < 30) for c in calls)
    # An exact oracle gives the same scores as the built-in enumeration
    enumerated = OAUAI(pgm_evaluator=mn_evaluator, k=4)(probs, query_mask, other, other)
    assert torch.allclose(mn_evaluator(out), mn_evaluator(enumerated))


def test_knn_fractional_k(mn_evaluator):
    n = mn_evaluator.num_variables
    probs = torch.rand(3, n, device=DEVICE)
    query_mask = torch.rand(3, n, device=DEVICE) > 0.5
    query_mask[2] = False  # a query with no query variables
    # Non-query variables hold evidence values, as after apply_evidence
    probs = torch.where(query_mask, probs, (probs > 0.5).float())
    out = KNearestDiscretizer(pgm_evaluator=mn_evaluator, k=0.1)(probs, query_mask)
    assert out.shape == probs.shape
    assert torch.all((out == 0) | (out == 1))


# --- loss ---


def test_entropy_penalty(mn_evaluator):
    n = mn_evaluator.num_variables
    q = torch.rand(4, n, device=DEVICE)
    query_mask = torch.rand(4, n, device=DEVICE) > 0.5
    # Equation (2): -alpha * sum_{j in Q} [q log q + (1 - q) log(1 - q)], averaged over the batch
    expected = -2.0 * ((q * q.log() + (1 - q) * (1 - q).log()) * query_mask).sum(-1).mean()
    assert torch.allclose(entropy_penalty(q, query_mask, alpha=2.0), expected, rtol=1e-5)
    # Non-query variables do not contribute, whatever their value
    q_changed = torch.where(query_mask, q, torch.full_like(q, 0.5))
    assert torch.allclose(entropy_penalty(q_changed, query_mask), entropy_penalty(q, query_mask))
    # alpha = 0 and discrete values give (almost) no penalty
    assert entropy_penalty(q, query_mask, alpha=0.0) == 0
    assert entropy_penalty((q > 0.5).float(), query_mask) < 1e-3


# --- trainer ---


def test_trainer_restores_best_validation_epoch(mn_evaluator):
    n = mn_evaluator.num_variables
    train_loader = DataLoader(TensorDataset(*[t.cpu() for t in make_batch(n, 16)]), batch_size=8)
    val_loader = DataLoader(TensorDataset(*[t.cpu() for t in make_batch(n, 8, seed=1)]), batch_size=8)
    model = make_model(n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=0)
    trainer = SelfSupervisedTrainer(
        model=model,
        pgm_evaluator=mn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer=optimizer,
        device=DEVICE,
        entropy_weight=1.0,
    )
    trainer.fit(train_loader, num_epochs=4, val_dataloader=val_loader, scheduler=scheduler)
    assert trainer.best_epoch is not None
    assert trainer.evaluate(val_loader) == pytest.approx(trainer.best_val_loss, rel=1e-5)


# --- ITSELF ---


@pytest.mark.parametrize("which", ["knn", "oauai"])
def test_itself_with_mask_discretizers(mn_evaluator, which):
    n = mn_evaluator.num_variables
    if which == "knn":
        discretizer = KNearestDiscretizer(pgm_evaluator=mn_evaluator, k=0.1)
    else:
        discretizer = OAUAI(pgm_evaluator=mn_evaluator, k=0.02)  # 2^4 candidates per query
    engine = ITSELF_Engine(
        model=make_model(n),
        pgm_evaluator=mn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer_cls=torch.optim.Adam,
        discretizer=discretizer,
        refinement_lr=1e-3,
        refinement_steps=2,
        device=DEVICE,
        entropy_weight=1.0,
    )
    batch = make_batch(n)
    out = engine.run(batch)["final_assignments"].to(DEVICE)
    evidence_data, evidence_mask = batch[0], batch[1]
    assert torch.all((out == 0) | (out == 1))
    assert torch.equal(out[evidence_mask].float(), evidence_data[evidence_mask])


def test_itself_single_query_with_batchnorm(mn_evaluator):
    n = mn_evaluator.num_variables
    engine = ITSELF_Engine(
        model=make_model(n, use_batchnorm=True),
        pgm_evaluator=mn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer_cls=torch.optim.Adam,
        discretizer=ThresholdDiscretizer(),
        refinement_lr=1e-3,
        refinement_steps=3,
        device=DEVICE,
    )
    batch = make_batch(n, num_samples=4)
    loader = DataLoader(TensorDataset(*[t.cpu() for t in batch]), batch_size=1)
    assert engine.run(loader)["final_assignments"].shape == (4, n)


def test_itself_early_stopping(mn_evaluator):
    n = mn_evaluator.num_variables
    num_calls = []

    def counting_loss(preds, evaluator):
        num_calls.append(1)
        return mpe_log_likelihood_loss(preds, evaluator)

    def run(tol):
        num_calls.clear()
        ITSELF_Engine(
            model=make_model(n),
            pgm_evaluator=mn_evaluator,
            loss_fn=counting_loss,
            optimizer_cls=torch.optim.SGD,
            discretizer=ThresholdDiscretizer(),
            refinement_lr=0.0,  # loss never changes, so any tol stops immediately
            refinement_steps=10,
            device=DEVICE,
            tol=tol,
            patience=2,
        ).run(make_batch(n))
        return len(num_calls)

    assert run(None) == 10
    assert run(1e-6) == 3


# --- MMAP ---


@pytest.fixture(scope="module")
def spn_evaluator():
    json_path = Path(__file__).parent.parent.parent / "examples/networks/spn/nltcs/spn.json"
    return SumProductNetwork(json_file=str(json_path), device=DEVICE)


def make_mmap_batch(num_vars, num_samples=4, seed=0):
    evidence_data, evidence_mask, query_mask, _ = make_batch(num_vars, num_samples, seed)
    # Half of the query variables become unobserved (to be summed out)
    gen = torch.Generator().manual_seed(seed + 1)
    unobs_mask = query_mask & (torch.rand(num_samples, num_vars, generator=gen) > 0.5).to(DEVICE)
    return evidence_data, evidence_mask, query_mask & ~unobs_mask, unobs_mask


def test_mark_unobserved_marginalizes(spn_evaluator):
    n = spn_evaluator.num_var
    x = torch.randint(0, 2, (3, n), device=DEVICE).float()
    unobs = torch.zeros(3, n, dtype=torch.bool, device=DEVICE)
    unobs[:, :2] = True
    # Brute-force sum over the two unobserved variables
    terms = []
    for a in (0.0, 1.0):
        for b in (0.0, 1.0):
            xi = x.clone()
            xi[:, 0], xi[:, 1] = a, b
            terms.append(spn_evaluator(xi))
    expected = torch.logsumexp(torch.stack(terms), dim=0)
    got = evaluate_log_likelihood(spn_evaluator, x, unobs)
    assert torch.allclose(got, expected, atol=1e-4)
    assert torch.equal(evaluate_log_likelihood(spn_evaluator, x), spn_evaluator(x))


def test_mmap_rejected_for_markov_network(mn_evaluator):
    n = mn_evaluator.num_variables
    x = torch.rand(2, n, device=DEVICE)
    unobs = torch.zeros(2, n, dtype=torch.bool, device=DEVICE)
    assert torch.equal(evaluate_log_likelihood(mn_evaluator, x, unobs), mn_evaluator(x))
    unobs[:, 0] = True
    with pytest.raises(ValueError, match="only supported for probabilistic circuits"):
        evaluate_log_likelihood(mn_evaluator, x, unobs)
    batch = make_mmap_batch(n)
    trainer = SelfSupervisedTrainer(
        model=make_model(n),
        pgm_evaluator=mn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer=torch.optim.Adam(make_model(n).parameters()),
        device=DEVICE,
    )
    with pytest.raises(ValueError):
        trainer.step(batch)


def test_mmap_training_uses_marginal_likelihood(spn_evaluator):
    n = spn_evaluator.num_var
    evidence_data, evidence_mask, query_mask, unobs_mask = make_mmap_batch(n, num_samples=8)
    model = make_model(n)
    trainer = SelfSupervisedTrainer(
        model=model,
        pgm_evaluator=spn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        device=DEVICE,
        entropy_weight=1.0,
    )
    batch = (evidence_data, evidence_mask, query_mask, unobs_mask)
    model.eval()
    with torch.no_grad():
        preds = apply_evidence(
            torch.sigmoid(model(*batch)), evidence_data, evidence_mask
        )
        expected = -evaluate_log_likelihood(spn_evaluator, preds, unobs_mask).mean()
        expected = expected + entropy_penalty(preds, query_mask, 1.0)
        assert torch.allclose(trainer._compute_loss(batch), expected)
    loss = trainer.step(batch)
    assert torch.isfinite(torch.tensor(loss))
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize("which", ["threshold", "knn", "oauai", "best_of"])
def test_mmap_itself_and_discretizers(spn_evaluator, which):
    n = spn_evaluator.num_var
    discretizers = {
        "threshold": lambda: ThresholdDiscretizer(),
        "knn": lambda: KNearestDiscretizer(pgm_evaluator=spn_evaluator, k=8),
        "oauai": lambda: OAUAI(pgm_evaluator=spn_evaluator, k=3),
        "best_of": lambda: BestOfDiscretizer(
            spn_evaluator,
            [OAUAI(pgm_evaluator=spn_evaluator, k=3), KNearestDiscretizer(spn_evaluator, k=8)],
        ),
    }
    batch = make_mmap_batch(n)
    out = ITSELF_Engine(
        model=make_model(n),
        pgm_evaluator=spn_evaluator,
        loss_fn=mpe_log_likelihood_loss,
        optimizer_cls=torch.optim.Adam,
        discretizer=discretizers[which](),
        refinement_lr=1e-3,
        refinement_steps=2,
        device=DEVICE,
    ).run(batch)["final_assignments"].to(DEVICE)
    assert torch.all((out == 0) | (out == 1))
    assert torch.isfinite(evaluate_log_likelihood(spn_evaluator, out.float(), batch[3])).all()


def test_best_of_discretizer_picks_best_per_query(mn_evaluator):
    n = mn_evaluator.num_variables
    evidence_data, evidence_mask, query_mask, unobs_mask = make_batch(n, num_samples=6)
    probs = apply_evidence(torch.rand(6, n, device=DEVICE), evidence_data, evidence_mask)
    oauai = OAUAI(pgm_evaluator=mn_evaluator, k=4)
    knn = KNearestDiscretizer(pgm_evaluator=mn_evaluator, k=16)
    combined = BestOfDiscretizer(mn_evaluator, [oauai, knn])
    args = (probs, query_mask, evidence_mask, unobs_mask)
    best = mn_evaluator(combined(*args))
    expected = torch.maximum(mn_evaluator(oauai(*args)), mn_evaluator(knn(*args)))
    assert torch.allclose(best, expected)
