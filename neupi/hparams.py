"""
Default hyperparameters for NeuPI's methods.

`default_hparams(method, model_type)` returns a dict with one entry per component:

- "model": keyword arguments of `MLP` (besides the output size and the embedding),
- "embedding": keyword arguments of the embedding (`DiscreteEmbedder`),
- "optimizer": keyword arguments of `torch.optim.Adam`,
- "scheduler": keyword arguments of `torch.optim.lr_scheduler.ReduceLROnPlateau`,
- "trainer": keyword arguments of the trainer (`SelfSupervisedTrainer` or `GUIDETrainer`),
- "fit": keyword arguments of `fit` (`num_epochs`, `selection`),
- "batch_size": the mini-batch size,
- "itself": keyword arguments of `ITSELF_Engine` (refinement at test time),
- "discretizers": keyword arguments of `KNearestDiscretizer` ("knn") and `OAUAI` ("oauai").

Fine-tune them for your model and query ratio on validation queries.
"""

import copy
from typing import Any, Dict

_HIDDEN = {
    # Network sizes from the papers: one hidden layer for graphical models and NAMs,
    # three for probabilistic circuits
    "pgm": [512],
    "nam": [512],
    "pc": [128, 256, 512],
}

_COMMON: Dict[str, Any] = {
    "optimizer": {"lr": 1e-3},
    "scheduler": {"factor": 0.9},
    "trainer": {"entropy_weight": -10.0},
    "fit": {"num_epochs": 100, "selection": "assignment"},
    "batch_size": 512,
    "itself": {"refinement_lr": 1e-3, "refinement_steps": 100, "entropy_weight": -1.0},
    "discretizers": {"knn": {"k": 1024}, "oauai": {"k": 0.25, "max_k": 200}},
}

_METHODS: Dict[str, Dict[str, Any]] = {
    # Self-supervised training (SelfSupervisedTrainer) with a DiscreteEmbedder
    "ssmp": {"model": {"dropout_rate": 0.0}, "embedding": {}},
    # Teacher-student training (GUIDETrainer)
    "guide": {
        "model": {"dropout_rate": 0.0},
        "embedding": {},
        "optimizer": {"lr": 3e-3},
        "trainer": {"entropy_weight": -10.0, "teacher_steps": 100, "teacher_lr": 1e-3},
    },
}


def default_hparams(method: str, model_type: str = "pgm") -> Dict[str, Any]:
    """
    Returns default hyperparameters for a method as a new dict (see the module docstring).

    Args:
        method (str): "ssmp" (`SelfSupervisedTrainer`) or "guide" (`GUIDETrainer`).
        model_type (str): The probabilistic model: "pgm" (`ProbabilisticGraphicalModel`),
            "pc" (`SumProductNetwork`) or "nam" (`MADE`); sets the network size.
            Defaults to "pgm".

    Example:
        >>> cfg = neupi.default_hparams("guide")
        >>> model = neupi.MLP(output_size=n, embedding=neupi.DiscreteEmbedder(n), **cfg["model"])
        >>> optimizer = torch.optim.Adam(model.parameters(), **cfg["optimizer"])
        >>> trainer = neupi.GUIDETrainer(model, pgm, neupi.mpe_log_likelihood_loss,
        ...                              optimizer, **cfg["trainer"])
    """
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(_METHODS)}.")
    if model_type not in _HIDDEN:
        raise ValueError(f"Unknown model_type {model_type!r}; choose from {sorted(_HIDDEN)}.")
    config = copy.deepcopy(_COMMON)
    config["model"] = {"hidden_sizes": list(_HIDDEN[model_type])}
    for key, value in copy.deepcopy(_METHODS[method]).items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config
