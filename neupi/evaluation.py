"""Evaluating solutions: scoring assignments, exact inference on small queries, comparisons."""

from typing import Callable, Dict, Optional, Tuple, Union

import torch

from neupi.utils.pgm_utils import evaluate_log_likelihood


def _evaluator_device(pgm_evaluator: Callable, fallback: torch.device) -> torch.device:
    if isinstance(pgm_evaluator, torch.nn.Module):
        for tensor in list(pgm_evaluator.parameters()) + list(pgm_evaluator.buffers()):
            return tensor.device
    device = getattr(pgm_evaluator, "device", None)
    return torch.device(device) if device is not None else fallback


@torch.no_grad()
def score_assignments(
    pgm_evaluator: Callable,
    assignments: torch.Tensor,
    unobs_mask: Optional[torch.Tensor] = None,
    batch_size: int = 4096,
) -> torch.Tensor:
    """
    Log-likelihoods of complete assignments under the probabilistic model.

    With unobserved variables (Marginal MAP) the score is log p(e, q), with the unobserved
    variables summed out, which needs an evaluator that can marginalize (`SumProductNetwork`).

    Args:
        pgm_evaluator: The probabilistic model, e.g. `ProbabilisticGraphicalModel`.
        assignments (torch.Tensor): Shape (num_examples, num_vars), e.g. the
            `final_assignments` returned by an inference engine.
        unobs_mask (torch.Tensor, optional): Mask of the unobserved variables.
        batch_size (int): Number of assignments scored at once.

    Returns:
        torch.Tensor: Shape (num_examples,), on the CPU.
    """
    device = _evaluator_device(pgm_evaluator, assignments.device)
    scores = []
    for start in range(0, assignments.shape[0], batch_size):
        batch = assignments[start : start + batch_size].to(device=device, dtype=torch.float32)
        batch_unobs = None if unobs_mask is None else unobs_mask[start : start + batch_size].to(device)
        scores.append(evaluate_log_likelihood(pgm_evaluator, batch, batch_unobs).float().cpu())
    return torch.cat(scores) if scores else torch.zeros(0)


@torch.no_grad()
def exact_inference(
    pgm_evaluator: Callable,
    evidence_data: torch.Tensor,
    evidence_mask: torch.Tensor,
    query_mask: torch.Tensor,
    unobs_mask: Optional[torch.Tensor] = None,
    max_query_vars: int = 20,
    batch_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solves each query exactly by scoring every assignment of its query variables.

    This is the exact MPE (or, with unobserved variables, Marginal MAP) solution, for
    checking neural solvers on queries with few query variables: the cost is
    2^(number of query variables) evaluations per query.

    Args:
        pgm_evaluator: The probabilistic model.
        evidence_data, evidence_mask, query_mask, unobs_mask: The queries, in NeuPI's
            four-tensor format, shape (num_examples, num_vars).
        max_query_vars (int): Refuse queries with more query variables than this.
        batch_size (int): Number of assignments scored at once.

    Returns:
        (assignments, log_likelihoods): The best assignment of each query, shape
        (num_examples, num_vars), and its score, shape (num_examples,), both on the CPU.
        Unobserved variables are 0 in the assignments.
    """
    evidence_mask = evidence_mask.bool().cpu()
    query_mask = query_mask.bool().cpu()
    unobs_mask = None if unobs_mask is None else unobs_mask.bool().cpu()
    most = int(query_mask.sum(dim=1).max()) if query_mask.numel() else 0
    if most > max_query_vars:
        raise ValueError(
            f"A query has {most} query variables; exact inference scores 2^{most} assignments. "
            f"Raise max_query_vars to allow it."
        )

    base = torch.where(evidence_mask, evidence_data.cpu().float(), torch.zeros(()))
    best_assignments = base.clone()
    best_scores = torch.empty(base.shape[0])
    for i in range(base.shape[0]):
        query_indices = query_mask[i].nonzero().flatten()
        k = len(query_indices)
        best_score = float("-inf")
        for start in range(0, 2**k, batch_size):
            codes = torch.arange(start, min(start + batch_size, 2**k))
            # Row r sets the query variables to the binary digits of code r
            bits = ((codes.unsqueeze(1) >> torch.arange(k)) & 1).float()
            candidates = base[i].expand(len(codes), -1).clone()
            candidates[:, query_indices] = bits
            candidate_unobs = None if unobs_mask is None else unobs_mask[i].expand_as(candidates)
            scores = score_assignments(pgm_evaluator, candidates, candidate_unobs, batch_size)
            score, idx = torch.max(scores, dim=0)
            if score.item() > best_score:
                best_score = score.item()
                best_assignments[i] = candidates[idx]
        best_scores[i] = best_score
    return best_assignments, best_scores


def compare_assignments(
    pgm_evaluator: Callable,
    assignments: Dict[str, torch.Tensor],
    reference: Optional[Union[str, torch.Tensor]] = None,
    unobs_mask: Optional[torch.Tensor] = None,
    evidence_data: Optional[torch.Tensor] = None,
    evidence_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Dict[str, Union[float, torch.Tensor]]]:
    """
    Scores the assignments of several methods on the same queries.

    Args:
        pgm_evaluator: The probabilistic model.
        assignments (dict): Method name -> assignments of shape (num_examples, num_vars).
        reference (str or torch.Tensor, optional): The name of the reference method in
            `assignments`, or reference log-likelihoods of shape (num_examples,), e.g. from
            `exact_inference` or another solver.
        unobs_mask (torch.Tensor, optional): Mask of the unobserved variables (Marginal MAP).
        evidence_data, evidence_mask (torch.Tensor, optional): If given, every method's
            assignments are checked to agree with the evidence.

    Returns:
        dict: Method name -> {
            "log_likelihoods": per-query scores,
            "mean_log_likelihood": their mean,
            and with a reference: "mean_gap", the mean of (reference - score), and
            "mean_relative_gap", the mean of (reference - score) / |reference|.
            Gaps are positive where the method scores below the reference.
        }
    """
    scores = {name: score_assignments(pgm_evaluator, a, unobs_mask) for name, a in assignments.items()}
    if evidence_data is not None and evidence_mask is not None:
        mask = evidence_mask.bool().cpu()
        for name, a in assignments.items():
            if (a.cpu().float()[mask] != evidence_data.cpu().float()[mask]).any():
                raise ValueError(f"The assignments of {name!r} do not agree with the evidence.")
    if isinstance(reference, str):
        if reference not in scores:
            raise ValueError(f"reference {reference!r} is not one of {list(scores)}.")
        reference_scores: Optional[torch.Tensor] = scores[reference]
    else:
        reference_scores = None if reference is None else torch.as_tensor(reference).float().cpu()

    results: Dict[str, Dict[str, Union[float, torch.Tensor]]] = {}
    for name, s in scores.items():
        entry: Dict[str, Union[float, torch.Tensor]] = {
            "log_likelihoods": s,
            "mean_log_likelihood": s.mean().item(),
        }
        if reference_scores is not None:
            if reference_scores.shape != s.shape:
                raise ValueError("reference must have one score per query.")
            gap = reference_scores - s
            entry["mean_gap"] = gap.mean().item()
            entry["mean_relative_gap"] = (gap / reference_scores.abs().clamp_min(1e-12)).mean().item()
        results[name] = entry
    return results
