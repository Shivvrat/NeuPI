import abc
import math
import numbers

import torch


class BaseDiscretizer(torch.nn.Module):
    """
    Abstract base class for all discretizers.

    A discretizer that searches may accept an optional ``deadline`` keyword (an absolute
    `time.monotonic()` value, see `neupi.utils.timing`); the inference engines pass one
    when they have a time limit, and the discretizer then returns its best incumbent once
    the deadline passes. Discretizers without the keyword are called without it.
    """

    @abc.abstractmethod
    def __call__(
        self,
        prob_outputs: torch.Tensor,
        query_mask: torch.Tensor = None,
        evidence_mask: torch.Tensor = None,
        unobs_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError


def validate_k(k, max_k=None):
    """
    Checks that ``k`` is a positive integer count or a fraction in (0, 1), and that
    ``max_k``, if given, is a positive integer count.

    Integer types such as numpy integers are accepted as counts. Floats must be strictly
    below 1 so that ``k=1`` (one) and a fraction cannot be confused.
    """
    if isinstance(k, bool) or not isinstance(k, numbers.Real):
        raise TypeError("k must be an integer count or a float fraction.")
    if isinstance(k, numbers.Integral):
        if k < 1:
            raise ValueError("An integer k must be at least 1.")
    elif not 0.0 < k < 1.0:
        raise ValueError(
            "A float k is a fraction of the query variables and must be in (0, 1); "
            "use an integer for a fixed count."
        )
    if max_k is not None:
        if isinstance(max_k, bool) or not isinstance(max_k, numbers.Integral) or max_k < 1:
            raise ValueError("max_k must be a positive integer.")
    return k


def resolve_k(k, num_query_vars: int, max_k=None) -> int:
    """
    Returns the k to use for one query.

    An integer k is used as given. A float k in (0, 1) is a fraction of the query's
    query variables: ceil(k * num_query_vars), and at least 1 when the query has
    any query variables. The result is then capped at ``max_k`` if given, e.g.
    ``k=0.25, max_k=200`` gives min(ceil(|Q| / 4), 200).
    """
    if isinstance(k, numbers.Integral):
        k = int(k)
    elif num_query_vars == 0:
        return 0
    else:
        k = max(1, math.ceil(k * num_query_vars))
    if max_k is not None:
        k = min(k, int(max_k))
    return k
