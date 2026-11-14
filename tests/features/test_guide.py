"""Tests for GUIDE training, the entropy bonus and the logits passed to discretizers."""

from pathlib import Path

import numpy as np
import torch
from neupi import (
    MLP,
    DiscreteEmbedder,
    GUIDETrainer,
    ProbabilisticGraphicalModel,
    QueryDataset,
    make_query_masks,
    mpe_log_likelihood_loss,
)
from test_robustness import random_model, reference_log_likelihood, write_uai
from torch.utils.data import DataLoader

MN_DIR = Path(__file__).parent.parent.parent / "examples/networks/mn"
GRID = str(MN_DIR / "Grids_17.uai")


def higher_order_model(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    scopes, tables = random_model(rng, 12, [1, 1, 2, 2, 3, 3, 3, 2, 1])
    return write_uai(tmp_path / "m.uai", 12, scopes, tables), scopes, tables


def test_guide_trainer(tmp_path):
    torch.manual_seed(0)
    path, scopes, tables = higher_order_model(tmp_path)
    pgm = ProbabilisticGraphicalModel(path)
    samples = torch.randint(0, 2, (40, 12)).float()
    masks = make_query_masks(40, 12, query_ratio=0.5, per_example=True, seed=0)
    dataset = QueryDataset(samples, *masks)
    model = MLP([16], 12, DiscreteEmbedder(12))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    trainer = GUIDETrainer(
        model, pgm, mpe_log_likelihood_loss, optimizer, teacher_steps=5, entropy_weight=0.1
    )
    # The teacher starts from different weights than the student
    assert not torch.equal(trainer.teacher.output_layer.weight, model.output_layer.weight)
    trainer.fit(
        DataLoader(dataset, batch_size=16, shuffle=True),
        3,
        val_dataloader=DataLoader(dataset, batch_size=16),
        verbose=False,
    )

    assert len(trainer.history) == 3 and trainer.best_epoch is not None
    assert torch.isfinite(trainer.db_loss).all()
    # Each database entry keeps its query's evidence and is scored correctly
    evidence = masks[0]
    assert torch.equal(trainer.db_assignments[evidence], samples[evidence])
    ll = reference_log_likelihood(trainer.db_assignments.numpy().astype(np.int64), scopes, tables)
    np.testing.assert_allclose(-trainer.db_loss.numpy(), ll, rtol=1e-4, atol=1e-4)
    # T <- S at the end
    for a, b in zip(trainer.teacher.parameters(), model.parameters()):
        assert torch.equal(a, b)


def test_k_nearest_from_logits_matches_brute_force():
    import itertools

    from neupi.discretize.kn import k_nearest_from_logits

    rng = np.random.default_rng(0)
    for _ in range(100):
        n = int(rng.integers(1, 8))
        logits = rng.normal(0, 2, n)
        k = int(rng.integers(1, 2**n + 2))
        q = 1 / (1 + np.exp(-logits))
        everything = np.array(list(itertools.product([0, 1], repeat=n)))
        best = np.sort(np.abs(q - everything).sum(1))[: min(k, 2**n)]
        found = k_nearest_from_logits(logits, k)
        np.testing.assert_allclose(np.sort(np.abs(q - found).sum(1)), best, atol=1e-9)


def test_saturated_outputs_keep_their_order():
    from neupi.discretize.kn import k_nearest_from_logits

    # In float32, sigmoid rounds all of these to exactly 0 or 1; the least confident
    # (|logit| = 90) flips first
    logits = np.array([150.0, -120.0, 90.0, 200.0])
    assert torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).tolist() == [1.0, 0.0, 1.0, 1.0]
    candidates = k_nearest_from_logits(logits, 3)
    assert candidates[0].tolist() == [1, 0, 1, 1]
    assert candidates[1].tolist() == [1, 0, 0, 1]
    assert candidates[2].tolist() == [1, 1, 1, 1]


def test_engines_pass_logits_to_discretizers():
    from neupi import OAUAI, SinglePassInferenceEngine

    seen = {}

    class Oracle:
        def __call__(self, assignment, free):
            seen["free"] = sorted(free.tolist())
            return assignment

    class Fixed(torch.nn.Module):
        def forward(self, x, e, q, u):
            return torch.tensor([[150.0, -120.0, 90.0, 200.0, -95.0]]).expand(x.shape[0], -1)

    class Flat(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0])

    discretizer = OAUAI(Flat(), k=2, oracle=Oracle())
    engine = SinglePassInferenceEngine(Fixed(), discretizer)
    x = torch.zeros(1, 5)
    query = torch.ones(1, 5, dtype=torch.bool)
    engine.run((x, ~query, query, torch.zeros_like(query)))
    # The two smallest |logit|, although all probabilities are exactly 0 or 1
    assert seen["free"] == [2, 4]


def test_guide_skips_batches_without_query_variables(tmp_path):
    path, _, _ = higher_order_model(tmp_path)
    pgm = ProbabilisticGraphicalModel(path)
    samples = torch.randint(0, 2, (8, 12)).float()
    evidence = torch.ones(8, 12, dtype=torch.bool)
    dataset = QueryDataset(samples, evidence, ~evidence)
    model = MLP([8], 12, DiscreteEmbedder(12))
    trainer = GUIDETrainer(
        model, pgm, mpe_log_likelihood_loss, torch.optim.Adam(model.parameters()), teacher_steps=2
    )
    trainer.fit(DataLoader(dataset, batch_size=4), 1, verbose=False)
    assert trainer.history[0]["train_loss"] == 0.0


def test_negative_entropy_weight_is_a_bonus():
    from neupi import ITSELF_Engine, SelfSupervisedTrainer, ThresholdDiscretizer, entropy_penalty

    torch.manual_seed(0)
    q = torch.rand(4, 6)
    mask = torch.rand(4, 6) > 0.5
    assert torch.allclose(entropy_penalty(q, mask, -2.0), -entropy_penalty(q, mask, 2.0))
    pgm = ProbabilisticGraphicalModel(GRID)
    n = pgm.num_variables
    model = MLP([16], n, DiscreteEmbedder(n))
    SelfSupervisedTrainer(
        model, pgm, mpe_log_likelihood_loss, torch.optim.Adam(model.parameters()), "cpu",
        entropy_weight=-1.0,
    )
    ITSELF_Engine(
        model, pgm, mpe_log_likelihood_loss, torch.optim.Adam, ThresholdDiscretizer(),
        refinement_lr=1e-3, refinement_steps=1, device="cpu", entropy_weight=-1.0,
    )
