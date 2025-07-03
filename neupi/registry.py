# neupi/registry.py
"""A lightweight registry for plug-in components."""
import functools
from typing import Any, Callable, Dict

_REGISTRY: Dict[str, Dict[str, Callable[..., Any]]] = {}


def register(kind: str) -> Callable[[type], type]:
    """A decorator to register a class in a given category (e.g., 'trainer')."""
    if not isinstance(kind, str) or not kind:
        raise TypeError("The 'kind' argument must be a non-empty string.")

    def _wrap(cls: type) -> type:
        if kind not in _REGISTRY:
            _REGISTRY[kind] = {}
        if cls.__name__ in _REGISTRY[kind]:
            # This helps catch accidental duplicate class names
            raise ValueError(f"'{cls.__name__}' is already registered in '{kind}'.")

        _REGISTRY[kind][cls.__name__] = cls
        return cls

    return _wrap


def get(kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
    """Instantiate a class from the registry by its kind and name."""
    try:
        return _REGISTRY[kind][name](*args, **kwargs)
    except KeyError:
        raise ValueError(
            f"'{name}' is not a registered '{kind}'. "
            f"Available: {list(_REGISTRY.get(kind, {}).keys())}"
        )
