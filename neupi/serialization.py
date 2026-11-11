"""Saving and loading trained neural solvers."""

import inspect
import os
from typing import Any, Dict, Optional, Union

import torch

from neupi.registry import _REGISTRY
from neupi.registry import get as factory

FORMAT = "neupi-solver"
FORMAT_VERSION = 1

PathLike = Union[str, "os.PathLike[str]"]


def _registered_name(obj: Any, kind: str) -> str:
    name = type(obj).__name__
    if _REGISTRY.get(kind, {}).get(name) is not type(obj):
        raise ValueError(
            f"{name} is not a registered '{kind}', so it cannot be rebuilt when loading. "
            f"Register it with @neupi.registry.register('{kind}')."
        )
    return name


def _plain(value: Any, where: str) -> Any:
    """Checks that a config value can be stored and loaded with torch.load(weights_only=True)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v, where) for v in value]
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _plain(v, where) for k, v in value.items()}
    raise ValueError(f"{where} has a value of type {type(value).__name__}, which cannot be saved.")


def _config(obj: Any, skip: tuple = ()) -> Dict[str, Any]:
    """
    The constructor arguments of `obj`: `obj.get_config()` if it has one, else the attributes
    named like the constructor's parameters.
    """
    name = type(obj).__name__
    if hasattr(obj, "get_config"):
        config = obj.get_config()
    else:
        config = {}
        for param in list(inspect.signature(type(obj).__init__).parameters.values())[1:]:
            if param.name in skip or param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            if not hasattr(obj, param.name):
                raise ValueError(
                    f"{name} does not store its argument {param.name!r}, so it cannot be "
                    f"rebuilt when loading; give it a get_config() method returning its arguments."
                )
            config[param.name] = getattr(obj, param.name)
    return _plain(config, name)


def save_solver(path: PathLike, model: torch.nn.Module, metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Saves a trained neural solver (e.g. an `MLP` and its embedding) to a file.

    The file holds the weights and the arguments that rebuild the network and its embedding,
    so `load_solver(path)` returns a ready-to-use model without the original code that built
    it. The model and embedding classes must be registered (all of NeuPI's are).

    Args:
        path: The file to write, e.g. "solver.pt".
        model (torch.nn.Module): The trained network. Its `embedding` attribute, if any, is
            saved with it.
        metadata (dict, optional): Extra information to store, e.g. the model file and
            query split the solver was trained for. Values must be numbers, strings,
            booleans, None, or lists/dicts of these.
    """
    embedding = getattr(model, "embedding", None)
    checkpoint = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "neupi_version": _neupi_version(),
        "model": {"class": _registered_name(model, "nn_model"), "config": _config(model, skip=("embedding",))},
        "embedding": None
        if embedding is None
        else {"class": _registered_name(embedding, "embedding"), "config": _config(embedding)},
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "metadata": _plain(metadata or {}, "metadata"),
    }
    torch.save(checkpoint, path)


def load_solver(path: PathLike, device: Union[str, torch.device] = "cpu", return_metadata: bool = False):
    """
    Loads a neural solver saved with `save_solver`.

    The file is read with ``torch.load(weights_only=True)``, which does not run code from it.

    Args:
        path: The file written by `save_solver`.
        device: The device to put the model on.
        return_metadata (bool): Also return the `metadata` dict given to `save_solver`.

    Returns:
        The model, in eval mode (and the metadata if `return_metadata`).
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != FORMAT:
        raise ValueError(f"{path} is not a solver saved with neupi.save_solver.")
    if checkpoint["format_version"] > FORMAT_VERSION:
        raise ValueError(
            f"{path} was saved by neupi {checkpoint['neupi_version']} in a newer format; "
            "upgrade neupi to load it."
        )
    kwargs = dict(checkpoint["model"]["config"])
    if checkpoint["embedding"] is not None:
        kwargs["embedding"] = factory(
            "embedding", checkpoint["embedding"]["class"], **checkpoint["embedding"]["config"]
        )
    model = factory("nn_model", checkpoint["model"]["class"], **kwargs)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    if return_metadata:
        return model, checkpoint["metadata"]
    return model


def _neupi_version() -> str:
    from neupi import __version__

    return __version__
