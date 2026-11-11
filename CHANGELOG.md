# Changelog

All notable changes to NeuPI. Versions follow [semantic versioning](https://semver.org/);
each release is on [PyPI](https://pypi.org/project/neupi/) and
[GitHub](https://github.com/Shivvrat/NeuPI/releases).

## 1.4.0 (2026-11-11)

### New
- Wheels for macOS (Apple silicon) and Windows (x86_64) besides Linux x86_64, and a source
  distribution on PyPI, so other platforms can build from source. Every wheel is installed
  and tested in CI before release.
- `make_query_masks` and `QueryDataset` build queries in the four-tensor format: query and
  unobserved variables given by index or as a random share of the variables, shared by all
  queries or drawn per query. `QueryDataset` hides the values of non-evidence variables.
- `save_solver` / `load_solver` save a trained network with its embedding and the
  arguments that rebuild them, plus optional metadata; loading uses
  `torch.load(weights_only=True)`.
- `score_assignments`, `exact_inference` (exact MPE or Marginal MAP by enumeration, for
  queries with few query variables) and `compare_assignments` (mean log-likelihood and the
  gap to a reference per method, with an optional evidence check).
- `SelfSupervisedTrainer.fit` takes `verbose` (default True, the current output) and a
  per-epoch `callback` that can stop training; the per-epoch losses, learning rate and
  duration are kept in `trainer.history`.

### Changed
- Requires Python 3.9 or newer (3.8 is end of life).
- `cython` and `setuptools` are no longer runtime dependencies (they are only needed to
  build).
- Building without OpenMP is supported (`NEUPI_OPENMP=0`; the default on macOS), with the
  UAI parser's parallel loops then running serially.

### Fixed
- The UAI parser used the C type `long` for int64 arrays, which is 32-bit on Windows.

### Development
- CI runs ruff and mypy; `pip install -e ".[dev]"` installs the pinned versions.
- Added `CONTRIBUTING.md`, this changelog and issue templates.

## 1.3.1 (2026-11-05)

### New
- Time limits for inference. `SinglePassInferenceEngine` and `ITSELF_Engine` take a
  `time_limit` in seconds per batch of queries (batch size 1 for a per-query limit),
  covering the forward pass, the ITSELF refinement and the discretizer, plus per-stage limits
  (`refinement_time_limit`, `discretizer_time_limit`) in seconds or as a percentage of
  `time_limit` (e.g. `"30%"`). When a budget runs out the best incumbent is returned, at
  worst the thresholded assignment. `run` also reports `elapsed_seconds`,
  `time_limit_reached` and (ITSELF) `refinement_steps_run` per batch.
- ITSELF keeps time back for a search discretizer so refinement cannot use the whole budget:
  the `discretizer_time_limit` if given, otherwise `discretizer_time_fraction` of the budget
  (default 10%).
- `KNearestDiscretizer`, `OAUAI` and `BestOfDiscretizer` take their own `time_limit` and an
  optional `deadline` from the engines; discretizers without a `deadline` argument still work.

### Fixed
- `KNearestDiscretizer` accepts float64/float16 probabilities.
- `MLP` without an embedding raises a clear `ValueError`.
- `ITSELF_Engine` creates its optimizer once at construction, so PyTorch's lazy imports are
  not charged to the first batch's time budget.
- Docs: front page, API reference and the `docs` extra.

## 1.3.0 (2026-11-04)

### Breaking / behaviour changes
- `SumProductNetwork` scores are exact: Bernoulli leaves compute `log(x*p + (1-x)*(1-p))`
  and sum weights use `log(w)` without smoothing. Scores differ from 1.2.0.
- `SumProductNetwork` raises on graphs with no single root, unreachable nodes or cycles.

### New
- `ProbabilisticGraphicalModel` reads Markov and Bayesian networks in UAI format, detected
  from the header (`network_type`). `MarkovNetwork` remains as an alias.
- `MADE` works as an evaluator: `forward` returns the log-likelihood, `get_logits` the
  network outputs.

### Fixed
- Pairwise models without unary factors (or without pairwise factors) no longer crash.
- Non-binary UAI models raise a clear error.
- SPN children are read from the JSON links, so any node order and DAGs work.
- Evaluators accept bool/int and unbatched input, check the input width and return scores on
  the input's device; SPN evaluates half precision in float32; MADE accepts any dtype.
- `apply_evidence` and `DiscreteEmbedder` treat integer/float masks as boolean masks.
- The trainer handles batches of one sample with BatchNorm.
- `factory()` no longer reports constructor errors as unregistered names.

## 1.2.0 (2026-11-01)

- Marginal MAP: unobserved variables are marginalized in the trainer, ITSELF and the
  discretizers (probabilistic circuits only).
- `BestOfDiscretizer` (e.g. OAUAI | k-Nearest from SINE).
- `KNearestDiscretizer` / `OAUAI`: `k` may be a fraction of the query variables, with
  `max_k`; OAUAI takes an optional exact oracle.
- ITSELF: early stopping (`tol`, `patience`), `freeze_batchnorm` and the entropy penalty.
- Trainer: entropy penalty, validation-based model selection and an LR scheduler.
- `MLP`: BatchNorm is off by default, as in the papers. Models saved with 1.1.0 defaults
  need `use_batchnorm=True` to load.
- Wheels are built without `-march=native` and `-ffast-math`, so they run on any x86_64 CPU.

## 1.1.0 and 1.0.3 (2025-11-17)

- Fixed the number of variables computed by `SumProductNetwork`.

## 1.0.0 – 1.0.2 (2025-07)

- First releases.
