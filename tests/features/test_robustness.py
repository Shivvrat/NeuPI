"""Robustness tests: unusual but valid models and inputs must work, invalid ones must raise clearly.

The models are generated in a temporary directory, so no model files are added to the repository.
"""

import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from neupi import (
    MADE,
    MLP,
    OAUAI,
    BestOfDiscretizer,
    DiscreteEmbedder,
    IdentityEmbedding,
    ITSELF_Engine,
    KNearestDiscretizer,
    MarkovNetwork,
    ProbabilisticGraphicalModel,
    SelfSupervisedTrainer,
    SinglePassInferenceEngine,
    SumProductNetwork,
    ThresholdDiscretizer,
    mpe_log_likelihood_loss,
)
from neupi.utils.pgm_utils import apply_evidence
from torch.utils.data import DataLoader, TensorDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPN_DIR = Path(__file__).parent.parent.parent / "examples/networks/spn"


# ----------------------------------------------------------------------------------------
# Markov networks
# ----------------------------------------------------------------------------------------


def write_uai(path, num_vars, scopes, tables, kind="MARKOV"):
    lines = [kind, str(num_vars), " ".join(["2"] * num_vars), str(len(scopes))]
    lines += [" ".join(map(str, [len(s)] + list(s))) for s in scopes]
    for t in tables:
        lines += ["", str(len(t)), " ".join(f"{v:.6g}" for v in t)]
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def reference_log_likelihood(x, scopes, tables):
    """Sum of log table entries, read in UAI order (first scope variable most significant)."""
    out = np.zeros(len(x))
    for s, t in zip(scopes, tables):
        idx = np.zeros(len(x), dtype=np.int64)
        for v in s:
            idx = idx * 2 + x[:, v]
        out += np.log(np.maximum(np.asarray(t), 1e-10))[idx]
    return out


def random_model(rng, num_vars, scope_sizes, zero_prob=0.0):
    scopes, tables = [], []
    for size in scope_sizes:
        scope = list(rng.choice(num_vars, size=size, replace=False))
        table = rng.uniform(0.1, 5.0, 2**size)
        table[rng.random(2**size) < zero_prob] = 0.0
        scopes.append(scope)
        tables.append(table.tolist())
    return scopes, tables


# (num_vars, factor scope sizes): pairwise without unary factors (as in the chook planted
# and Wishart models), unary only, mixed pairwise, higher order without pairwise factors,
# mixed higher order, a single variable, and variables that are in no factor
MODEL_SHAPES = {
    "pairwise_no_unary": (6, [2] * 8),
    "unary_only": (5, [1] * 5),
    "pairwise_and_unary": (6, [1] * 3 + [2] * 6),
    "higher_order_no_pairwise": (6, [1, 1, 3, 3, 4]),
    "higher_order_mixed": (7, [1, 2, 2, 3, 4, 5]),
    "single_variable": (1, [1]),
    "isolated_variables": (6, [2, 2]),
}


@pytest.mark.parametrize("shape", list(MODEL_SHAPES))
@pytest.mark.parametrize("kind", ["MARKOV", "BAYES"])
def test_markov_network_matches_reference(tmp_path, shape, kind):
    num_vars, sizes = MODEL_SHAPES[shape]
    rng = np.random.default_rng(len(shape))
    scopes, tables = random_model(rng, num_vars, sizes, zero_prob=0.1)
    mn = ProbabilisticGraphicalModel(write_uai(tmp_path / "m.uai", num_vars, scopes, tables, kind), device=DEVICE)
    assert mn.num_variables == num_vars

    x = np.array(list(itertools.product([0, 1], repeat=num_vars)))
    got = mn(torch.tensor(x, dtype=torch.float32, device=DEVICE)).double().cpu().numpy()
    np.testing.assert_allclose(got, reference_log_likelihood(x, scopes, tables), rtol=1e-5, atol=1e-4)

    # Relaxed assignments give finite scores and gradients, even with zero table entries
    x_soft = torch.rand(4, num_vars, device=DEVICE, requires_grad=True)
    scores = mn(x_soft)
    scores.sum().backward()
    assert torch.isfinite(scores).all() and torch.isfinite(x_soft.grad).all()


@pytest.mark.parametrize("kind", ["MARKOV", "BAYES", "markov", "bayes"])
def test_network_type_is_detected_from_the_header(tmp_path, kind):
    scopes, tables = [[0], [0, 1]], [[0.3, 0.7], [0.9, 0.1, 0.2, 0.8]]
    path = write_uai(tmp_path / "m.uai", 2, scopes, tables, kind)
    pgm = ProbabilisticGraphicalModel(path)
    assert pgm.network_type == kind.upper()
    x = np.array(list(itertools.product([0, 1], repeat=2)))
    np.testing.assert_allclose(pgm(torch.tensor(x)).numpy(), reference_log_likelihood(x, scopes, tables), rtol=1e-5)
    if kind == "BAYES":  # a Bayesian network is normalized
        assert torch.isclose(pgm(torch.tensor(x)).exp().sum(), torch.tensor(1.0))
    # The earlier name is the same class and reads the same files
    old = MarkovNetwork(path)
    assert isinstance(old, ProbabilisticGraphicalModel) and old.network_type == kind.upper()
    assert torch.equal(old(torch.tensor(x)), pgm(torch.tensor(x)))


def test_markov_network_without_unary_factors_matches_ones_unary_copy(tmp_path):
    """A model without unary factors scores like the same model with all-ones unary factors."""
    rng = np.random.default_rng(0)
    scopes, tables = random_model(rng, 5, [2] * 6)
    mn = ProbabilisticGraphicalModel(write_uai(tmp_path / "a.uai", 5, scopes, tables))
    assert mn.pgm.univariate_tables is None
    ref = ProbabilisticGraphicalModel(
        write_uai(tmp_path / "b.uai", 5, scopes + [[v] for v in range(5)], tables + [[1.0, 1.0]] * 5)
    )
    x = torch.tensor(list(itertools.product([0, 1], repeat=5)), dtype=torch.float32)
    assert torch.allclose(mn(x), ref(x))


def test_markov_network_file_format_variants(tmp_path):
    """Unsorted scopes, CRLF line endings, tables split across lines and a constant factor."""
    text = "MARKOV\r\n3\r\n2 2 2\r\n3\r\n2 2 0\r\n0\r\n3 1 0 2\r\n4\r\n1 2\r\n3 4\r\n1 7.0\r\n8 1 2 3 4 5 6 7 8\r\n"
    path = tmp_path / "v.uai"
    path.write_bytes(text.encode())
    mn = ProbabilisticGraphicalModel(str(path))
    x = np.array(list(itertools.product([0, 1], repeat=3)))
    scopes = [[2, 0], [], [1, 0, 2]]
    tables = [[1, 2, 3, 4], [7.0], list(range(1, 9))]
    got = mn(torch.tensor(x, dtype=torch.float32)).double().numpy()
    np.testing.assert_allclose(got, reference_log_likelihood(x, scopes, tables), rtol=1e-5)


def test_markov_network_input_types(tmp_path):
    rng = np.random.default_rng(1)
    for sizes in ([2, 2, 1], [3, 2]):
        scopes, tables = random_model(rng, 4, sizes)
        mn = ProbabilisticGraphicalModel(write_uai(tmp_path / f"m{len(sizes)}.uai", 4, scopes, tables))
        x = torch.tensor([[1, 0, 1, 1], [0, 0, 1, 0]])
        expected = mn(x.float())
        for dtype in (torch.bool, torch.int64, torch.float64):
            assert torch.allclose(mn(x.to(dtype)).float(), expected)
        assert torch.allclose(mn(x[0].float()), expected[:1])  # an unbatched assignment
        with pytest.raises(ValueError, match="shape"):
            mn(torch.zeros(2, 5))


def test_markov_network_rejects_unsupported_models(tmp_path):
    non_binary = tmp_path / "nb.uai"
    non_binary.write_text("MARKOV\n2\n3 2\n1\n2 0 1\n6 1 2 3 4 5 6\n")
    with pytest.raises(ValueError, match="binary"):
        ProbabilisticGraphicalModel(str(non_binary))
    other = tmp_path / "o.uai"
    other.write_text("FOO\n1\n2\n1\n1 0\n2 1 1\n")
    with pytest.raises(ValueError, match="MARKOV or BAYES"):
        ProbabilisticGraphicalModel(str(other))


# ----------------------------------------------------------------------------------------
# Sum-product networks
# ----------------------------------------------------------------------------------------


def reference_spn(data, x):
    nodes = {n["id"]: n for n in data["nodes"]}
    children = {}
    for link in sorted(data["links"], key=lambda l: l["idx"]):
        children.setdefault(link["target"], []).append(link["source"])
    root = next(i for i in nodes if all(l["source"] != i for l in data["links"]))

    def value(i, row, cache):
        if i not in cache:
            node = nodes[i]
            if node["class"] == "Bernoulli":
                v, p = row[node["scope"][0]], node["params"]["p"]
                cache[i] = 0.0 if np.isnan(v) else math.log(v * p + (1 - v) * (1 - p))
            elif node["class"] == "Product":
                cache[i] = sum(value(c, row, cache) for c in children[i])
            else:
                terms = [math.log(w) + value(c, row, cache) for w, c in zip(node["weights"], children[i])]
                cache[i] = float(np.logaddexp.reduce(terms))
        return cache[i]

    return np.array([value(root, row, {}) for row in x])


def spn_from_data(tmp_path, data, name="spn.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return SumProductNetwork(str(path), device=DEVICE)


def shuffled_copy(data, seed=0):
    """The same SPN with renamed node ids and nodes and links in random order."""
    rnd = random.Random(seed)
    ids = [n["id"] for n in data["nodes"]]
    new_ids = dict(zip(ids, rnd.sample(range(1000, 1000 + 3 * len(ids)), len(ids))))
    nodes = [dict(n, id=new_ids[n["id"]]) for n in data["nodes"]]
    links = [dict(l, source=new_ids[l["source"]], target=new_ids[l["target"]]) for l in data["links"]]
    rnd.shuffle(nodes)
    rnd.shuffle(links)
    return {"nodes": nodes, "links": links}


# A DAG in which two product nodes share a leaf
DAG_SPN = {
    "nodes": [
        {"class": "Sum", "id": 0, "scope": [0, 1], "weights": [0.3, 0.7]},
        {"class": "Product", "id": 1, "scope": [0, 1]},
        {"class": "Product", "id": 2, "scope": [0, 1]},
        {"class": "Bernoulli", "id": 3, "scope": [0], "params": {"p": 0.2}},
        {"class": "Bernoulli", "id": 4, "scope": [1], "params": {"p": 0.9}},
        {"class": "Bernoulli", "id": 5, "scope": [0], "params": {"p": 0.6}},
    ],
    "links": [
        {"idx": 0, "source": 1, "target": 0},
        {"idx": 1, "source": 2, "target": 0},
        {"idx": 0, "source": 3, "target": 1},
        {"idx": 1, "source": 4, "target": 1},
        {"idx": 0, "source": 5, "target": 2},
        {"idx": 1, "source": 4, "target": 2},
    ],
}


@pytest.mark.parametrize("variant", ["nltcs", "nltcs_shuffled", "plants", "plants_shuffled", "dag"])
def test_spn_matches_reference(tmp_path, variant):
    if variant == "dag":
        data = DAG_SPN
    else:
        data = json.loads((SPN_DIR / variant.split("_")[0] / "spn.json").read_text())
        if variant.endswith("shuffled"):
            data = shuffled_copy(data)
    spn = spn_from_data(tmp_path, data)
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, (8, spn.num_variables)).astype(float)
    x[0, :2] = np.nan  # marginalized
    got = spn(torch.tensor(x, dtype=torch.float32, device=DEVICE)).double().cpu().numpy()
    np.testing.assert_allclose(got, reference_spn(data, x), atol=1e-4)


def test_spn_dag_sums_to_one(tmp_path):
    spn = spn_from_data(tmp_path, DAG_SPN)
    x = torch.tensor(list(itertools.product([0, 1], repeat=2)), dtype=torch.float32)
    assert torch.isclose(spn(x).exp().sum().cpu(), torch.tensor(1.0), atol=1e-5)
    assert torch.isclose(spn(torch.full((1, 2), float("nan"))).exp().cpu(), torch.tensor(1.0), atol=1e-5)


def test_spn_rejects_invalid_graphs(tmp_path):
    two_roots = json.loads(json.dumps(DAG_SPN))
    two_roots["links"] = two_roots["links"][1:]
    with pytest.raises(ValueError, match="root"):
        spn_from_data(tmp_path, two_roots, "a.json")
    bad_weights = json.loads(json.dumps(DAG_SPN))
    bad_weights["nodes"][0]["weights"] = [1.0]
    with pytest.raises(ValueError, match="weights"):
        spn_from_data(tmp_path, bad_weights, "b.json")
    spn = spn_from_data(tmp_path, DAG_SPN, "c.json")
    with pytest.raises(ValueError, match="NaN"):
        spn(-torch.ones(1, 2))
    with pytest.raises(ValueError, match="shape"):
        spn(torch.zeros(1, 3))


# ----------------------------------------------------------------------------------------
# MADE
# ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_vars", [1, 4])
def test_made_is_a_normalized_evaluator(num_vars):
    torch.manual_seed(0)
    made = MADE(num_vars, [16, 16])
    x = torch.tensor(list(itertools.product([0, 1], repeat=num_vars)), dtype=torch.float32)
    scores = made(x)
    assert scores.shape == (2**num_vars,)
    assert torch.isclose(scores.exp().sum(), torch.tensor(1.0), atol=1e-5)
    assert made.get_logits(x).shape == x.shape


# ----------------------------------------------------------------------------------------
# Masks, devices and batches through the whole pipeline
# ----------------------------------------------------------------------------------------


def test_integer_and_float_masks_mean_the_same_as_bool_masks():
    pred = torch.full((2, 3), 0.5)
    evidence = torch.ones(2, 3)
    mask = torch.tensor([[True, False, False], [False, False, True]])
    expected = apply_evidence(pred, evidence, mask)
    emb = DiscreteEmbedder(3)
    expected_emb = emb(evidence, mask, ~mask, torch.zeros_like(mask))
    for dtype in (torch.long, torch.uint8, torch.float32):
        m = mask.to(dtype)
        assert torch.equal(apply_evidence(pred, evidence, m), expected)
        assert torch.equal(emb(evidence.long(), m, 1 - m.long(), torch.zeros_like(m)), expected_emb)
    # An unbatched query
    assert torch.equal(emb(evidence[0], mask[0], ~mask[0], torch.zeros(3, dtype=torch.bool)), expected_emb[0])


def make_query_batch(num_vars, batch_size, pattern, mmap=False, seed=0):
    gen = torch.Generator().manual_seed(seed)
    evidence_data = torch.randint(0, 2, (batch_size, num_vars), generator=gen)
    if pattern == "all_query":
        evidence_mask = torch.zeros(batch_size, num_vars, dtype=torch.bool)
    elif pattern == "all_evidence":
        evidence_mask = torch.ones(batch_size, num_vars, dtype=torch.bool)
    elif pattern == "one_query":
        evidence_mask = torch.ones(batch_size, num_vars, dtype=torch.bool)
        evidence_mask[:, 0] = False
    else:
        evidence_mask = torch.rand(batch_size, num_vars, generator=gen) > 0.5
    unobs_mask = torch.zeros(batch_size, num_vars, dtype=torch.bool)
    if mmap:
        unobs_mask = ~evidence_mask & (torch.rand(batch_size, num_vars, generator=gen) > 0.5)
    query_mask = ~evidence_mask & ~unobs_mask
    return evidence_data, evidence_mask, query_mask, unobs_mask


def evaluators():
    root = Path(__file__).parent.parent.parent / "examples/networks"
    yield "mn", ProbabilisticGraphicalModel(str(root / "mn/Grids_17.uai"), device="cpu")
    yield "spn", SumProductNetwork(str(SPN_DIR / "nltcs/spn.json"), device="cpu")
    torch.manual_seed(0)
    yield "made", MADE(10, [16])


@pytest.fixture(scope="module", params=["mn", "spn", "made"])
def evaluator(request):
    return dict(evaluators())[request.param]


@pytest.mark.parametrize("pattern", ["mixed", "all_query", "all_evidence", "one_query"])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("integer_inputs", [False, True])
def test_pipeline_handles_edge_case_batches(evaluator, pattern, batch_size, integer_inputs):
    """Training, single-pass and ITSELF inference with every discretizer. The evaluator stays
    on the CPU while the network runs on DEVICE (CUDA when available)."""
    n = evaluator.num_variables
    mmap = isinstance(evaluator, SumProductNetwork) and pattern == "mixed"
    data = make_query_batch(n, batch_size, pattern, mmap=mmap)
    if integer_inputs:
        data = (data[0], *(m.long() for m in data[1:]))
    else:
        data = (data[0].float(), *data[1:])

    torch.manual_seed(0)
    model = MLP([8], n, embedding=DiscreteEmbedder(n), use_batchnorm=True)
    trainer = SelfSupervisedTrainer(
        model, evaluator, mpe_log_likelihood_loss, torch.optim.Adam(model.parameters()),
        device=DEVICE, entropy_weight=0.1,
    )
    # batch_size=2 leaves a last batch with one sample when batch_size is 1 or 3
    trainer.fit(DataLoader(TensorDataset(*data), batch_size=2), num_epochs=1)

    discretizers = {
        "threshold": ThresholdDiscretizer(),
        "knearest": KNearestDiscretizer(evaluator, k=8),
        "knearest_more_than_all": KNearestDiscretizer(evaluator, k=5000),
        "oauai": OAUAI(evaluator, k=2),
        "oauai_fraction": OAUAI(evaluator, k=0.5, max_k=3),
        "best_of": BestOfDiscretizer(evaluator, [OAUAI(evaluator, k=2), KNearestDiscretizer(evaluator, k=4)]),
    }
    evidence_mask = data[1].bool()
    evidence = data[0].long()
    for name, discretizer in discretizers.items():
        out = SinglePassInferenceEngine(model, discretizer, device=DEVICE).run(data)["final_assignments"]
        assert out.shape == (batch_size, n), name
        assert ((out == 0) | (out == 1)).all(), name
        assert torch.equal(out[evidence_mask], evidence[evidence_mask]), name

    out = ITSELF_Engine(
        model, evaluator, mpe_log_likelihood_loss, torch.optim.Adam, discretizers["best_of"],
        refinement_lr=1e-2, refinement_steps=2, device=DEVICE, entropy_weight=0.1,
    ).run(data)["final_assignments"]
    assert out.shape == (batch_size, n)
    assert torch.equal(out[evidence_mask], evidence[evidence_mask])


def test_identity_embedding_accepts_integer_evidence():
    emb = IdentityEmbedding(3)
    model = MLP([4], 3, embedding=emb)
    out = model(torch.tensor([[1, 0, 1]]), *(torch.zeros(1, 3, dtype=torch.bool),) * 3)
    assert out.shape == (1, 3)


def test_spn_deterministic_leaves_and_precision(tmp_path):
    data = json.loads(json.dumps(DAG_SPN))
    data["nodes"][3]["params"]["p"] = 0.0  # deterministic leaves
    data["nodes"][4]["params"]["p"] = 1.0
    spn = spn_from_data(tmp_path, data)
    x = torch.tensor([[1.0, 0.0]], device=DEVICE, requires_grad=True)
    score = spn(x)
    score.sum().backward()
    assert torch.isfinite(score).all() and torch.isfinite(x.grad).all() and (x.grad != 0).all()
    # A summed-out variable contributes exactly log 1
    assert torch.allclose(spn(torch.full((1, 2), float("nan"), device=DEVICE)), torch.zeros(1, device=DEVICE), atol=1e-6)

    # Low-precision inputs are evaluated in float32
    ad = SumProductNetwork(str(SPN_DIR / "ad/spn.json"), device=DEVICE)
    xb = torch.randint(0, 2, (4, ad.num_variables), device=DEVICE).float()
    expected = ad(xb)
    for dtype in (torch.float16, torch.bfloat16):
        assert torch.allclose(ad(xb.to(dtype)).float(), expected, atol=1e-3)


def test_made_accepts_any_input_dtype():
    made = MADE(4, [8])
    x = torch.tensor([[1, 0, 1, 0]])
    expected = made(x.float())
    for dtype in (torch.bool, torch.int64, torch.float64, torch.float16):
        assert torch.allclose(made(x.to(dtype)).float(), expected, atol=1e-3)


def test_knearest_accepts_any_float_dtype(tmp_path):
    rng = np.random.default_rng(2)
    scopes, tables = random_model(rng, 5, [1, 2, 2, 2])
    pgm = ProbabilisticGraphicalModel(write_uai(tmp_path / "m.uai", 5, scopes, tables))
    knn = KNearestDiscretizer(pgm, k=4)
    probs = torch.rand(3, 5)
    query_mask = torch.ones(3, 5, dtype=torch.bool)
    expected = knn(probs, query_mask)
    for dtype in (torch.float64, torch.float16):
        out = knn(probs.to(dtype), query_mask)
        assert out.dtype == dtype and torch.equal(out.float(), expected)


def test_mlp_without_embedding_raises_clearly():
    with pytest.raises(ValueError, match="IdentityEmbedding"):
        MLP([8], 4, None)


def test_custom_evaluator_and_discretizer_plug_in():
    """A user-defined evaluator (any BaseProbModel with forward) and discretizer work
    with the trainer, both engines and the built-in discretizers."""
    from neupi.core.discretizer import BaseDiscretizer
    from neupi.core.model import BaseProbModel

    class Target(BaseProbModel):
        # log-likelihood that prefers the all-ones assignment
        def forward(self, x):
            return -((x - 1.0) ** 2).sum(dim=1)

    class FirstHalfDiscretizer(BaseDiscretizer):
        def __call__(self, prob_outputs, query_mask=None, evidence_mask=None, unobs_mask=None):
            return (prob_outputs >= 0.5).to(prob_outputs.dtype)

    n, batch = 6, 4
    data = make_query_batch(n, batch, "mixed")
    data = (data[0].float(), *data[1:])
    loader = DataLoader(TensorDataset(*data), batch_size=2)
    torch.manual_seed(0)
    model = MLP([16], n, embedding=DiscreteEmbedder(n))
    evaluator = Target()
    trainer = SelfSupervisedTrainer(model, evaluator, mpe_log_likelihood_loss, torch.optim.Adam(model.parameters(), 0.1))
    before = trainer.evaluate(loader)
    trainer.fit(loader, num_epochs=20)
    assert trainer.evaluate(loader) < before

    for discretizer in (FirstHalfDiscretizer(), KNearestDiscretizer(evaluator, k=3), OAUAI(evaluator, k=2)):
        out = SinglePassInferenceEngine(model, discretizer).run(loader)["final_assignments"]
        assert out.shape == (batch, n) and ((out == 0) | (out == 1)).all()
    out = ITSELF_Engine(
        model, evaluator, mpe_log_likelihood_loss, torch.optim.Adam, FirstHalfDiscretizer(), 0.1, 3
    ).run(loader)["final_assignments"]
    evidence_mask = data[1]
    assert torch.equal(out[evidence_mask], data[0].long()[evidence_mask])
    # The trained solver sets the query variables to 1, which the evaluator prefers
    assert (out[~evidence_mask] == 1).all()


# ----------------------------------------------------------------------------------------
# Time limits
# ----------------------------------------------------------------------------------------


class SlowEvaluator(torch.nn.Module):
    """An evaluator that takes `delay` seconds per call, preferring the all-ones assignment."""

    def __init__(self, delay):
        super().__init__()
        self.delay = delay
        self.calls = 0

    def forward(self, x):
        import time

        time.sleep(self.delay)
        self.calls += 1
        return -((x - 1.0) ** 2).sum(dim=1)


def _time_limit_setup(n=8, batch=3):
    data = make_query_batch(n, batch, "mixed", seed=3)
    data = (data[0].float(), *data[1:])
    torch.manual_seed(0)
    model = MLP([8], n, embedding=DiscreteEmbedder(n))
    return data, model


@pytest.mark.parametrize("make", [
    lambda pm: KNearestDiscretizer(pm, k=64, time_limit=0),
    lambda pm: OAUAI(pm, k=3, time_limit=0),
    lambda pm: BestOfDiscretizer(pm, [OAUAI(pm, k=3), KNearestDiscretizer(pm, k=64)], time_limit=0),
])
def test_discretizer_with_no_time_returns_thresholded_incumbent(make):
    pm = SlowEvaluator(0.0)
    probs = torch.rand(3, 8)
    query_mask = torch.ones(3, 8, dtype=torch.bool)
    out = make(pm)(probs, query_mask, ~query_mask, torch.zeros_like(query_mask))
    assert torch.equal(out, (probs >= 0.5).float())
    # A deadline from the caller is honoured the same way
    out = KNearestDiscretizer(pm, k=64)(probs, query_mask, deadline=0.0)
    assert torch.equal(out, (probs >= 0.5).float())


def test_discretizer_time_limit_keeps_best_incumbent():
    """With the clock running out mid-search, the result is still at least as good as
    thresholding, and the search stops within the budget."""
    import time

    pm = SlowEvaluator(0.05)
    probs = torch.rand(4, 8)
    query_mask = torch.ones(4, 8, dtype=torch.bool)
    knn = KNearestDiscretizer(pm, k=256, batch_size=16, time_limit=0.3)
    t0 = time.monotonic()
    out = knn(probs, query_mask)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5
    assert (pm(out) >= pm((probs >= 0.5).float())).all()
    assert torch.equal(out[probs >= 0.9], torch.ones_like(out[probs >= 0.9]))


def test_single_pass_time_limit():
    data, model = _time_limit_setup()
    pm = SlowEvaluator(0.0)
    engine = SinglePassInferenceEngine(model, KNearestDiscretizer(pm, k=64), time_limit=0)
    out = engine.run(data)
    assert out["time_limit_reached"].tolist() == [True]
    assert out["elapsed_seconds"].shape == (1,)
    thresholded = torch.where(data[1], data[0], (out["prob_outputs"] >= 0.5).float())
    assert torch.equal(out["final_assignments"], thresholded.int())
    evidence = data[1]
    assert torch.equal(out["final_assignments"][evidence], data[0].long()[evidence])

    unlimited = SinglePassInferenceEngine(model, KNearestDiscretizer(pm, k=64)).run(data)
    assert unlimited["time_limit_reached"].tolist() == [False]

    # A discretizer without a `deadline` argument still works with a time limit
    class Plain(ThresholdDiscretizer):
        pass

    out = SinglePassInferenceEngine(model, Plain(), time_limit=0).run(data)
    assert out["final_assignments"].shape == data[0].shape

    with pytest.raises(ValueError, match="time_limit"):
        SinglePassInferenceEngine(model, Plain(), time_limit=-1)


def test_itself_time_limits():
    data, model = _time_limit_setup()
    loader = DataLoader(TensorDataset(*data), batch_size=2)  # 2 batches

    def engine(**kw):
        return ITSELF_Engine(
            model, SlowEvaluator(0.0), mpe_log_likelihood_loss, torch.optim.Adam,
            ThresholdDiscretizer(), 1e-2, refinement_steps=5, **kw,
        )

    out = engine().run(loader)
    assert out["refinement_steps_run"].tolist() == [5, 5]
    assert out["time_limit_reached"].tolist() == [False, False]
    assert out["elapsed_seconds"].shape == (2,)

    out = engine(refinement_time_limit=0).run(loader)
    assert out["refinement_steps_run"].tolist() == [0, 0]
    assert out["time_limit_reached"].tolist() == [True, True]

    # The total budget cuts the refinement short but still produces a valid answer
    slow = ITSELF_Engine(
        model, SlowEvaluator(0.05), mpe_log_likelihood_loss, torch.optim.Adam,
        KNearestDiscretizer(SlowEvaluator(0.05), k=64), 1e-2, refinement_steps=50, time_limit=0.2,
    )
    out = slow.run(loader)
    assert (out["refinement_steps_run"] < 50).all()
    assert (out["elapsed_seconds"] < 1.5).all()
    evidence = data[1]
    assert torch.equal(out["final_assignments"][evidence], data[0].long()[evidence])


def test_itself_reserves_time_for_the_discretizer():
    """Refinement may use at most (1 - fraction) of the budget when the discretizer can use
    time, so the discretizer always gets to run."""
    data, model = _time_limit_setup(batch=2)
    loader = DataLoader(TensorDataset(*data), batch_size=2)

    def run(discretizer, **kw):
        pm = SlowEvaluator(0.02)
        engine = ITSELF_Engine(
            model, pm, mpe_log_likelihood_loss, torch.optim.Adam, discretizer, 1e-2,
            refinement_steps=1000, time_limit=0.5, **kw,
        )
        out = engine.run(loader)
        return out, pm.calls

    assert ITSELF_Engine(
        model, SlowEvaluator(0), mpe_log_likelihood_loss, torch.optim.Adam, ThresholdDiscretizer(), 1e-2, 1
    ).discretizer_time_fraction == 0.1

    reserved_scorer = SlowEvaluator(0.02)
    out, reserved_steps = run(KNearestDiscretizer(reserved_scorer, k=8, batch_size=1), discretizer_time_fraction=0.4)
    assert out["time_limit_reached"].tolist() == [True]

    # With the reserve disabled, refinement uses the whole budget and the discretizer only
    # returns its incumbent
    full_scorer = SlowEvaluator(0.02)
    out, full_steps = run(KNearestDiscretizer(full_scorer, k=8, batch_size=1), discretizer_time_fraction=0)
    assert reserved_steps < full_steps
    assert reserved_scorer.calls > full_scorer.calls
    assert full_scorer.calls <= 2  # at most the thresholded-vs-best comparison

    # Thresholding needs no time, so nothing is reserved for it
    out, threshold_steps = run(ThresholdDiscretizer(), discretizer_time_fraction=0.4)
    assert threshold_steps > reserved_steps

    with pytest.raises(ValueError, match="discretizer_time_fraction"):
        ITSELF_Engine(model, SlowEvaluator(0), mpe_log_likelihood_loss, torch.optim.Adam,
                      ThresholdDiscretizer(), 1e-2, 1, discretizer_time_fraction=1.0)


def test_stage_budgets_accept_seconds_or_percentages():
    from neupi.utils.timing import stage_time_limit, time_share

    assert stage_time_limit(None, 2.0, "x") is None
    assert stage_time_limit(0.3, None, "x") == 0.3
    assert stage_time_limit("25%", 2.0, "x") == 0.5
    assert stage_time_limit(" 12.5 % ", 2.0, "x") == 0.25
    assert time_share(0.1, "x") == 0.1 and time_share("10%", "x") == 0.1
    for bad in ("25", "abc%", "150%", -1):
        with pytest.raises(ValueError):
            stage_time_limit(bad, 2.0, "x")
    with pytest.raises(ValueError, match="time_limit must be set"):
        stage_time_limit("25%", None, "x")
    for bad in (1.0, "100%", -0.1, True):
        with pytest.raises(ValueError):
            time_share(bad, "x")

    data, model = _time_limit_setup()
    pm = SlowEvaluator(0.0)

    def itself(**kw):
        return ITSELF_Engine(model, pm, mpe_log_likelihood_loss, torch.optim.Adam,
                             ThresholdDiscretizer(), 1e-2, 3, **kw)

    engine = itself(time_limit=2.0, refinement_time_limit="70%", discretizer_time_limit="30%",
                    discretizer_time_fraction="20%")
    assert engine.refinement_time_limit == pytest.approx(1.4)
    assert engine.discretizer_time_limit == pytest.approx(0.6)
    assert engine.discretizer_time_fraction == pytest.approx(0.2)
    assert itself(refinement_time_limit=0.5).refinement_time_limit == 0.5
    with pytest.raises(ValueError, match="time_limit must be set"):
        itself(discretizer_time_limit="30%")

    # A 0% stage budget stops that stage immediately
    out = itself(time_limit=10.0, refinement_time_limit="0%").run(data)
    assert out["refinement_steps_run"].tolist() == [0]
    out = SinglePassInferenceEngine(model, KNearestDiscretizer(pm, k=64), time_limit=10.0,
                                    discretizer_time_limit="0%").run(data)
    assert out["time_limit_reached"].tolist() == [True]


@pytest.mark.parametrize("budget", [{"discretizer_time_limit": "40%"}, {"discretizer_time_limit": 0.2}])
def test_itself_reserves_the_given_discretizer_budget(budget):
    """A per-step discretizer budget is kept back from refinement, as seconds or a percentage."""
    data, model = _time_limit_setup(batch=2)
    loader = DataLoader(TensorDataset(*data), batch_size=2)

    def run(**kw):
        pm, scorer = SlowEvaluator(0.02), SlowEvaluator(0.02)
        out = ITSELF_Engine(
            model, pm, mpe_log_likelihood_loss, torch.optim.Adam,
            KNearestDiscretizer(scorer, k=8, batch_size=1), 1e-2,
            refinement_steps=1000, time_limit=0.5, **kw,
        ).run(loader)
        return out, pm.calls, scorer.calls

    out, steps, scored = run(**budget)
    _, steps_without, scored_without = run(discretizer_time_fraction=0)
    assert out["time_limit_reached"].tolist() == [True]
    assert steps < steps_without and scored > scored_without
