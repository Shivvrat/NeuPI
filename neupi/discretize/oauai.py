import itertools
import numbers
from typing import Callable, Optional, Union

import torch

from neupi.core.discretizer import BaseDiscretizer, resolve_k, validate_k
from neupi.registry import register
from neupi.utils.pgm_utils import evaluate_log_likelihood


@register("discretizer_oauai")
class OAUAI(BaseDiscretizer):
    """
    Discretizes by searching over the k most uncertain query variables.

    This method thresholds the predictions, identifies the 'k' query variables with
    probabilities closest to 0.5, and re-assigns those k variables with the remaining
    variables fixed at their thresholded values. The result is kept only if it scores
    higher under the PGM evaluator than the thresholded assignment.

    By default the k variables are re-assigned by exhaustively scoring all 2^k binary
    assignments. An exact oracle (such as daoopt) can be supplied instead through
    ``oracle``.

    Args:
        pgm_evaluator (Callable): The PGM evaluator (e.g., ProbabilisticGraphicalModel) which acts
                                  as the scoring function.
        k (int or float): The number of most uncertain query variables to search over.
                          A float in (0, 1) is a fraction of each query's query
                          variables (ceil(k * num_query_vars), at least 1), so k varies
                          per query. An int k is capped at the number of query variables.
        threshold (float): The baseline threshold for comparison. Defaults to 0.5.
        max_k (int, optional): Upper limit on the per-query k. The SINE paper uses at most
                          1/4 of the query variables or 200, whichever is smaller:
                          ``k=0.25, max_k=200``.
        oracle (Callable, optional): Called as ``oracle(assignment, free_indices)`` for
                          each query, where ``assignment`` is the thresholded assignment
                          of shape (num_variables,) and ``free_indices`` is a LongTensor
                          of the k selected variables. It must return a full assignment of
                          shape (num_variables,) in which the free variables are set (for
                          example, the MPE of the free variables given all others). If
                          None, all 2^k assignments are enumerated.

    References
    ----------
    Arya, Shivvrat, Rahman, Tahrima, and Gogate, Vibhav Giridhar. SINE: Scalable MPE Inference for Probabilistic Graphical Models Using Advanced Neural Embeddings. Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS), 2025.
    """

    def __init__(
        self,
        pgm_evaluator: Callable,
        k: Union[int, float],
        threshold: float = 0.5,
        oracle: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_k: Optional[int] = None,
    ):
        super().__init__()
        validate_k(k, max_k)
        if oracle is None and isinstance(k, numbers.Integral) and min(k, max_k or k) > 10:
            print(f"Warning: k={k} may lead to slow performance (2^{k} evaluations per sample).")
        self.pgm_evaluator = pgm_evaluator
        self.k = k
        self.threshold = threshold
        self.oracle = oracle
        self.max_k = max_k
        self._warned_large_k = False

    @torch.no_grad()
    def __call__(
        self,
        prob_outputs: torch.Tensor,
        query_mask: torch.Tensor,
        evidence_mask: torch.Tensor,
        unobs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs the OAUAI discretization for each instance in the batch. .
        """
        num_examples, num_vars = prob_outputs.shape
        device = prob_outputs.device
        dtype = prob_outputs.dtype

        # All binary assignments for a given k, generated once per distinct k
        binary_assignments = {}

        # Get baseline assignments and scores from simple thresholding
        final_assignments = (prob_outputs >= self.threshold).to(dtype)
        final_scores = evaluate_log_likelihood(self.pgm_evaluator, final_assignments, unobs_mask)

        for i in range(num_examples):
            # Isolate the i-th sample
            sample_probs = prob_outputs[i]
            sample_query_mask = query_mask[i].bool()
            sample_unobs = None if unobs_mask is None else unobs_mask[i : i + 1]

            # Only query variables can be selected, so k cannot exceed their number
            num_query_vars = int(sample_query_mask.sum().item())
            k = min(resolve_k(self.k, num_query_vars, self.max_k), num_query_vars)
            if k == 0:
                continue

            # Find the k most uncertain query variables
            certainty = torch.abs(sample_probs - 0.5)
            # Mask out non-query variables so they are not selected
            certainty[~sample_query_mask] = float("inf")

            _, top_k_indices = torch.topk(certainty, k, largest=False)

            if self.oracle is not None:
                candidate = self.oracle(final_assignments[i].clone(), top_k_indices)
                candidate = torch.as_tensor(candidate, dtype=dtype, device=device).reshape(num_vars)
                candidate_score = evaluate_log_likelihood(
                    self.pgm_evaluator, candidate.unsqueeze(0), sample_unobs
                )[0]
                if candidate_score > final_scores[i]:
                    final_assignments[i] = candidate
                continue

            if k > 10 and not self._warned_large_k:
                print(f"Warning: k={k} may lead to slow performance (2^{k} evaluations per sample).")
                self._warned_large_k = True

            if k not in binary_assignments:
                binary_assignments[k] = torch.tensor(
                    list(itertools.product([0, 1], repeat=k)), dtype=dtype, device=device
                )
            binary_assignments_k = binary_assignments[k]
            num_candidates = binary_assignments_k.shape[0]

            # Create candidate assignments by modifying the thresholded base
            base_assignment = final_assignments[i].unsqueeze(0).repeat(num_candidates, 1)
            base_assignment[:, top_k_indices] = binary_assignments_k

            # Score all candidates and find the best one
            candidate_unobs = None if sample_unobs is None else sample_unobs.expand_as(base_assignment)
            candidate_scores = evaluate_log_likelihood(
                self.pgm_evaluator, base_assignment, candidate_unobs
            )
            best_candidate_score, best_candidate_idx = torch.max(candidate_scores, dim=0)

            # If the best candidate is better than the simple thresholded result, use it
            if best_candidate_score > final_scores[i]:
                final_assignments[i] = base_assignment[best_candidate_idx]

        return final_assignments
