"""Building queries: the evidence, query and unobserved masks and a Dataset of queries."""

from typing import Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

Indices = Union[Sequence[int], torch.Tensor]


def _split_count(ratio: float, num_vars: int, name: str) -> int:
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {ratio!r}.")
    count = round(ratio * num_vars)
    # A positive ratio gives at least one variable
    return max(count, 1) if ratio > 0 else 0


def _indices_mask(indices: Indices, num_vars: int, name: str) -> torch.Tensor:
    indices = torch.as_tensor(indices, dtype=torch.long).flatten()
    if indices.numel() and (indices.min() < 0 or indices.max() >= num_vars):
        raise ValueError(f"{name} must be variable indices in [0, {num_vars}).")
    mask = torch.zeros(num_vars, dtype=torch.bool)
    mask[indices] = True
    return mask


def make_query_masks(
    num_examples: int,
    num_vars: int,
    query_ratio: Optional[float] = None,
    unobs_ratio: float = 0.0,
    query_vars: Optional[Indices] = None,
    unobs_vars: Optional[Indices] = None,
    per_example: bool = False,
    seed: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits the variables into evidence, query and unobserved variables.

    The query variables are given either as indices (`query_vars`) or as a share of the
    variables chosen at random (`query_ratio`); the unobserved variables likewise
    (`unobs_vars` or `unobs_ratio`, default none, i.e. MPE). All other variables are evidence.
    A non-empty unobserved set makes the queries Marginal MAP, which needs an evaluator
    that can marginalize (`SumProductNetwork`).

    Args:
        num_examples (int): Number of queries (rows).
        num_vars (int): Number of variables of the probabilistic model.
        query_ratio (float, optional): Share of the variables that are query variables, in
            [0, 1]; a positive share gives at least one variable.
        unobs_ratio (float): Share of the variables that are unobserved. Defaults to 0.0.
        query_vars (sequence of int, optional): Indices of the query variables.
        unobs_vars (sequence of int, optional): Indices of the unobserved variables.
        per_example (bool): With ratios, draw a new random split for every query instead of
            one split shared by all queries. Defaults to False.
        seed (int, optional): Seed of the random split.
        device: Device of the returned masks.

    Returns:
        (evidence_mask, query_mask, unobs_mask): Boolean tensors of shape
        (num_examples, num_vars) that partition the variables.
    """
    if num_examples < 0 or num_vars <= 0:
        raise ValueError("num_examples must be non-negative and num_vars positive.")
    if (query_ratio is None) == (query_vars is None):
        raise ValueError("Give exactly one of query_ratio and query_vars.")
    if unobs_vars is not None and unobs_ratio:
        raise ValueError("Give at most one of unobs_ratio and unobs_vars.")

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    fixed_query = None if query_vars is None else _indices_mask(query_vars, num_vars, "query_vars")
    fixed_unobs = None if unobs_vars is None else _indices_mask(unobs_vars, num_vars, "unobs_vars")
    if fixed_query is not None and fixed_unobs is not None and (fixed_query & fixed_unobs).any():
        raise ValueError("query_vars and unobs_vars overlap.")

    num_query = None if query_ratio is None else _split_count(query_ratio, num_vars, "query_ratio")
    num_unobs = 0 if unobs_vars is not None else _split_count(unobs_ratio, num_vars, "unobs_ratio")
    taken = (int(fixed_query.sum()) if fixed_query is not None else num_query or 0) + (
        int(fixed_unobs.sum()) if fixed_unobs is not None else num_unobs
    )
    if taken > num_vars:
        raise ValueError("The query and unobserved variables together exceed num_vars.")

    def draw() -> Tuple[torch.Tensor, torch.Tensor]:
        query = fixed_query.clone() if fixed_query is not None else torch.zeros(num_vars, dtype=torch.bool)
        unobs = fixed_unobs.clone() if fixed_unobs is not None else torch.zeros(num_vars, dtype=torch.bool)
        # Randomly chosen sets come from the variables not already taken by fixed indices
        free = (~(query | unobs)).nonzero().flatten()
        free = free[torch.randperm(len(free), generator=generator)]
        if num_query is not None:
            query[free[:num_query]] = True
            free = free[num_query:]
        unobs[free[:num_unobs]] = True
        return query, unobs

    random_split = query_ratio is not None or (unobs_vars is None and num_unobs > 0)
    if per_example and random_split:
        pairs = [draw() for _ in range(num_examples)]
        query_mask = torch.stack([q for q, _ in pairs]) if pairs else torch.zeros(0, num_vars, dtype=torch.bool)
        unobs_mask = torch.stack([u for _, u in pairs]) if pairs else torch.zeros(0, num_vars, dtype=torch.bool)
    else:
        query, unobs = draw()
        query_mask = query.expand(num_examples, num_vars).clone()
        unobs_mask = unobs.expand(num_examples, num_vars).clone()
    evidence_mask = ~(query_mask | unobs_mask)
    return evidence_mask.to(device), query_mask.to(device), unobs_mask.to(device)


class QueryDataset(Dataset):
    """
    A Dataset of inference queries in NeuPI's four-tensor format.

    Each item is ``(evidence_data, evidence_mask, query_mask, unobs_mask)``, as expected by
    the trainers and inference engines. The values of the non-evidence variables are set to 0
    in `evidence_data`, so the true values of the query variables cannot leak into the
    network's input.

    Args:
        samples (torch.Tensor): Binary assignments of shape (num_examples, num_vars) that
            supply the evidence values, e.g. data the probabilistic model was learned from or
            samples from it. Random assignments (``torch.randint(0, 2, (n, num_vars))``) also
            work, but evidence drawn from the model is more representative.
        evidence_mask, query_mask (torch.Tensor): Masks of shape (num_examples, num_vars) or
            (num_vars,) (the same for every query).
        unobs_mask (torch.Tensor, optional): Mask of the unobserved variables (Marginal MAP).
            Defaults to none.

    Use `QueryDataset.from_split` to build the masks with `make_query_masks`.
    """

    def __init__(
        self,
        samples: torch.Tensor,
        evidence_mask: torch.Tensor,
        query_mask: torch.Tensor,
        unobs_mask: Optional[torch.Tensor] = None,
    ):
        samples = torch.as_tensor(samples)
        if samples.dim() != 2:
            raise ValueError("samples must have shape (num_examples, num_vars).")
        if not ((samples == 0) | (samples == 1)).all():
            raise ValueError("samples must be binary (0/1).")
        shape = samples.shape
        if unobs_mask is None:
            unobs_mask = torch.zeros(shape, dtype=torch.bool)

        def full(mask: torch.Tensor, name: str) -> torch.Tensor:
            mask = torch.as_tensor(mask).bool().cpu()
            try:
                return mask.expand(shape).clone()
            except RuntimeError:
                raise ValueError(f"{name} must have shape {tuple(shape)} or ({shape[1]},).") from None

        self.evidence_mask = full(evidence_mask, "evidence_mask")
        self.query_mask = full(query_mask, "query_mask")
        self.unobs_mask = full(unobs_mask, "unobs_mask")
        if (self.query_mask & (self.evidence_mask | self.unobs_mask)).any() or (
            self.evidence_mask & self.unobs_mask
        ).any():
            raise ValueError("The evidence, query and unobserved masks overlap.")
        self.evidence_data = torch.where(
            self.evidence_mask, samples.cpu().float(), torch.zeros((), dtype=torch.float32)
        )

    @classmethod
    def from_split(
        cls,
        samples: torch.Tensor,
        query_ratio: Optional[float] = None,
        unobs_ratio: float = 0.0,
        query_vars: Optional[Indices] = None,
        unobs_vars: Optional[Indices] = None,
        per_example: bool = False,
        seed: Optional[int] = None,
    ) -> "QueryDataset":
        """Builds the masks with `make_query_masks` (same arguments) for the given samples."""
        samples = torch.as_tensor(samples)
        if samples.dim() != 2:
            raise ValueError("samples must have shape (num_examples, num_vars).")
        masks = make_query_masks(
            samples.shape[0],
            samples.shape[1],
            query_ratio=query_ratio,
            unobs_ratio=unobs_ratio,
            query_vars=query_vars,
            unobs_vars=unobs_vars,
            per_example=per_example,
            seed=seed,
        )
        return cls(samples, *masks)

    def __len__(self) -> int:
        return self.evidence_data.shape[0]

    def __getitem__(self, idx):
        return (
            self.evidence_data[idx],
            self.evidence_mask[idx],
            self.query_mask[idx],
            self.unobs_mask[idx],
        )
