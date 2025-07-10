import itertools
from typing import Callable

import torch
from tqdm import tqdm

from neupi.core.discretizer import BaseDiscretizer
from neupi.registry import register


@register("discretizer_oauai")
class OAUAI(BaseDiscretizer):
    """
    Discretizes by performing an exhaustive search over the k most uncertain variables.

    This method identifies the 'k' query variables with probabilities closest to 0.5,
    generates all 2^k possible binary assignments for this subset, scores each one
    using the PGM evaluator, and selects the best assignment.

    Note: This is a naive oracle that looks at all 2^k possible assignments and selects the best one. Other oracles can also be used (such as daoopt).

    Args:
        pgm_evaluator (Callable): The PGM evaluator (e.g., MarkovNetwork) which acts
                                  as the scoring function.
        k (int): The number of most uncertain query variables to search over.
        threshold (float): The baseline threshold for comparison. Defaults to 0.5.

    References
    ----------
    Arya, Shivvrat, Rahman, Tahrima, and Gogate, Vibhav Giridhar. SINE: Scalable MPE Inference for Probabilistic Graphical Models Using Advanced Neural Embeddings. Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS), 2025.
    """

    def __init__(self, pgm_evaluator: Callable, k: int, threshold: float = 0.5):
        super().__init__()
        if k > 10:
            print(f"Warning: k={k} may lead to slow performance (2^{k} evaluations per sample).")
        self.pgm_evaluator = pgm_evaluator
        self.k = k
        self.threshold = threshold

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

        # Pre-generate all binary assignments for the k variables
        binary_assignments_k = torch.tensor(
            list(itertools.product([0, 1], repeat=self.k)), dtype=dtype, device=device
        )
        num_candidates = binary_assignments_k.shape[0]

        # Get baseline assignments and scores from simple thresholding
        final_assignments = (prob_outputs >= self.threshold).to(dtype)
        final_scores = self.pgm_evaluator(final_assignments)

        for i in tqdm(range(num_examples), desc="Uncertainty Discretization", leave=False):
            # Isolate the i-th sample
            sample_probs = prob_outputs[i]
            sample_query_mask = query_mask[i]

            # Find the k most uncertain query variables
            uncertainty = torch.abs(sample_probs - 0.5)
            # Mask out non-query variables so they are not selected
            uncertainty[~sample_query_mask] = float("inf")

            _, top_k_indices = torch.topk(uncertainty, self.k, largest=False)

            # Create candidate assignments by modifying the thresholded base
            base_assignment = final_assignments[i].unsqueeze(0).repeat(num_candidates, 1)
            base_assignment[:, top_k_indices] = binary_assignments_k

            # Score all candidates and find the best one
            candidate_scores = self.pgm_evaluator(base_assignment)
            best_candidate_score, best_candidate_idx = torch.max(candidate_scores, dim=0)

            # If the best candidate is better than the simple thresholded result, use it
            if best_candidate_score > final_scores[i]:
                final_assignments[i] = base_assignment[best_candidate_idx]

        return final_assignments
