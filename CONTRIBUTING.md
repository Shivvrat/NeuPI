# Contributing to NeuPI

Bug reports, fixes and new components are welcome.

## Reporting a bug

If NeuPI fails on your model or gives a wrong result, please
[open an issue](https://github.com/Shivvrat/NeuPI/issues/new/choose) even if you have a
workaround: other users probably hit the same problem. The bug report form asks for what we
need to reproduce it:

- the NeuPI, Python, PyTorch and OS versions (`python -c "import neupi, torch, sys; print(neupi.__version__, torch.__version__, sys.version)"`);
- the probabilistic model: its type (UAI Markov or Bayesian network, SPN JSON, MADE), number of
  variables and factor sizes, and if possible the file or a small model that shows the problem;
- the code you ran and the full traceback.

## Development setup

You need a C++ compiler (with OpenMP on Linux and Windows; macOS builds without it).

```bash
git clone https://github.com/Shivvrat/NeuPI.git
cd NeuPI
pip install -e ".[dev]"
```

`pip install -e .` compiles the two Cython extensions. After editing a `.pyx` file, rebuild
them with `python setup.py build_ext --inplace`. Building rewrites the committed generated
`.cpp`/`.html` files even if the `.pyx` did not change; revert those
(`git checkout -- '*.cpp' '*.html'`) unless you changed the `.pyx`.

## Checks

Run these before opening a pull request; CI runs the same:

```bash
ruff check neupi tests setup.py
mypy
pytest --ignore=tests/losses
```

`tests/losses` compares the evaluators with external tools (the Libra CLI and `deeprob-kit`);
it needs those installed and is not run in CI.

Tests generate their models in a temporary directory or use the small fixtures in
`examples/networks/`. Please don't add large model files to the repository.

## Code conventions

- All components exchange four tensors of shape `(batch, num_vars)`: `evidence_data`,
  `evidence_mask`, `query_mask`, `unobs_mask`. Neural models output logits; trainers and
  engines apply the sigmoid.
- New components subclass the base classes in `neupi/core/`, are registered with
  `@neupi.registry.register(kind)` and exported from `neupi/__init__.py`.
- A component that implements a method from a paper cites it in its docstring.
- Keep the public API backward compatible; if a change can't be, describe it in the pull
  request.
- Add a test for every fix or feature, and a line to the "Unreleased" section of
  `CHANGELOG.md`.

## Releases

Maintainers release by bumping the version in `pyproject.toml`, `neupi/__init__.py`,
`docs/source/conf.py` and `CITATION.cff` and moving the "Unreleased" changelog entries under
the new version. Pushing that commit to `main` publishes the release to PyPI and GitHub
after CI passes.
