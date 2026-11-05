"""Time budgets for inference.

A budget is a number of seconds; a deadline is an absolute `time.monotonic()` value (or None
for no limit). Engines turn their budgets into deadlines and hand them to the discretizers,
which stop searching when the deadline passes and return their best incumbent.
"""

import inspect
import time
from typing import Callable, Optional, Union


def validate_time_limit(time_limit: Optional[float], name: str = "time_limit") -> Optional[float]:
    if time_limit is None:
        return None
    if not isinstance(time_limit, (int, float)) or isinstance(time_limit, bool) or time_limit < 0:
        raise ValueError(f"{name} must be None or a non-negative number of seconds.")
    return float(time_limit)


def _parse_percent(value: str, name: str) -> float:
    """'25%' -> 25.0."""
    text = value.strip()
    if not text.endswith("%"):
        raise ValueError(f"{name} must be a number of seconds or a percentage such as '25%', got {value!r}.")
    try:
        return float(text[:-1])
    except ValueError:
        raise ValueError(f"{name} must be a number of seconds or a percentage such as '25%', got {value!r}.") from None


def stage_time_limit(
    value: Optional[Union[float, str]], total: Optional[float], name: str
) -> Optional[float]:
    """
    The budget in seconds for one stage of inference.

    `value` is None (no limit for the stage), a number of seconds, or a percentage of the
    total budget `total` as a string, e.g. ``"25%"``. A percentage needs a total budget.
    """
    if isinstance(value, str):
        percent = _parse_percent(value, name)
        if not 0.0 <= percent <= 100.0:
            raise ValueError(f"{name} must be a percentage between 0% and 100%, got {value!r}.")
        if total is None:
            raise ValueError(f"{name}={value!r} is a percentage of time_limit, so time_limit must be set.")
        return total * percent / 100.0
    return validate_time_limit(value, name)


def time_share(value: Union[float, str], name: str) -> float:
    """A share of a budget given as a fraction in [0, 1) or a percentage string such as '10%'."""
    if isinstance(value, str):
        share = _parse_percent(value, name) / 100.0
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        share = float(value)
    else:
        raise ValueError(f"{name} must be a fraction in [0, 1) or a percentage such as '10%'.")
    if not 0.0 <= share < 1.0:
        raise ValueError(f"{name} must be a fraction in [0, 1) or a percentage below 100%, got {value!r}.")
    return share


def deadline_from(time_limit: Optional[float], start: Optional[float] = None) -> Optional[float]:
    """The deadline for a budget of `time_limit` seconds starting at `start` (default: now)."""
    if time_limit is None:
        return None
    return (time.monotonic() if start is None else start) + time_limit


def min_deadline(*deadlines: Optional[float]) -> Optional[float]:
    """The earliest of the given deadlines, ignoring None (no limit)."""
    valid = [d for d in deadlines if d is not None]
    return min(valid) if valid else None


def expired(deadline: Optional[float]) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def accepts_deadline(fn: Callable) -> bool:
    """Whether `fn` (a discretizer) takes a `deadline` keyword argument."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "deadline" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def call_discretizer(discretizer: Callable, *args, deadline: Optional[float] = None):
    """Calls a discretizer, passing the deadline only if it supports one."""
    if deadline is not None and accepts_deadline(discretizer):
        return discretizer(*args, deadline=deadline)
    return discretizer(*args)
