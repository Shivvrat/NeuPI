from typing import Callable, Optional, Union

import torch

from neupi.core.discretizer import BaseDiscretizer, resolve_k, validate_k
from neupi.registry import register
from neupi.utils.pgm_utils import evaluate_log_likelihood

from .cython_kn.kn_binary_vectors import cython_process_assignments


@register("discretizer_knn")
class KNearestDiscretizer(BaseDiscretizer):
    """
    Finds k-nearest binary vectors for query variables using a scoring function.

    This method generates candidate binary assignments for query variables that are
    "close" to the continuous predictions, scores them using the PGM evaluator,
    and selects the best one.

    Args:
        pgm_evaluator (Callable): The PGM evaluator (e.g., MarkovNetwork) which acts
                                  as the scoring function.
        k (int or float): Number of nearest binary vectors to consider. A float in
                          (0, 1) is a fraction of each query's query variables
                          (ceil(k * num_query_vars), at least 1), so k varies per query.
        batch_size (int): Batch size for scoring candidate assignments.
        max_k (int, optional): Upper limit on the per-query k.


    References
    ----------
    Arya, Shivvrat, Rahman, Tahrima, and Gogate, Vibhav Giridhar. SINE: Scalable MPE Inference for Probabilistic Graphical Models Using Advanced Neural Embeddings. Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS), 2025.
    """

    def __init__(
        self,
        pgm_evaluator: Callable,
        k: Union[int, float],
        batch_size: int = 300,
        max_k: Optional[int] = None,
    ):
        super().__init__()
        validate_k(k, max_k)
        self.max_k = max_k
        self.pgm_evaluator = pgm_evaluator
        self.k = k
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(
        self,
        prob_outputs: torch.Tensor,
        query_mask: torch.Tensor,
        evidence_mask: torch.Tensor = None,
        unobs_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Selects k-nearest binary vectors for each continuous output from the NN and selects the best one.

        Args:
            prob_outputs (torch.Tensor): Continuous predictions from the network.
                Shape: (batch_size, num_variables).
            evidence_mask (torch.Tensor): Boolean mask for evidence variables.
            query_mask (torch.Tensor): Boolean mask for query variables.
            unobs_mask (torch.Tensor): Boolean mask for unobserved variables.

        Returns:
            torch.Tensor: The final discrete assignments.
                Shape: (batch_size, num_variables).
        """
        num_examples, num_vars = prob_outputs.shape
        device = prob_outputs.device
        dtype = prob_outputs.dtype

        query_indices = [torch.where(qm)[0] for qm in query_mask]
        thresholded_assignments = (prob_outputs >= 0.5).to(dtype)
        best_assignments = thresholded_assignments.clone()

        for i in range(num_examples):
            # Nothing to search for a query without query variables
            if len(query_indices[i]) == 0:
                continue
            query_probs_np = prob_outputs[i, query_indices[i]].detach().cpu().numpy()
            k = resolve_k(self.k, len(query_indices[i]), self.max_k)

            # Get k candidate binary assignments from the Cython helper
            _, candidate_np = cython_process_assignments(query_probs_np, k)
            candidates = torch.tensor(candidate_np, dtype=dtype, device=device)

            best_score = torch.tensor(float("-inf"), device=device)
            best_local_assignment = None

            # Score each candidate assignment in batches
            for j in range(0, candidates.shape[0], self.batch_size):
                batch_candidates = candidates[j : j + self.batch_size]

                # Create a batch of full assignments, each with a different candidate
                data_batch = best_assignments[i].unsqueeze(0).repeat(len(batch_candidates), 1)
                data_batch[:, query_indices[i]] = batch_candidates

                batch_unobs = None if unobs_mask is None else unobs_mask[i].expand_as(data_batch)
                scores = evaluate_log_likelihood(self.pgm_evaluator, data_batch, batch_unobs)
                max_score, max_idx = torch.max(scores, dim=0)

                if max_score > best_score:
                    best_score = max_score
                    best_local_assignment = batch_candidates[max_idx]

            best_assignments[i, query_indices[i]] = best_local_assignment

        # Compare with simple thresholding and return the better result
        thresholded_scores = evaluate_log_likelihood(
            self.pgm_evaluator, thresholded_assignments, unobs_mask
        )
        best_scores = evaluate_log_likelihood(self.pgm_evaluator, best_assignments, unobs_mask)

        final_assignments = torch.where(
            best_scores.unsqueeze(1) > thresholded_scores.unsqueeze(1),
            best_assignments,
            thresholded_assignments,
        )
        return final_assignments
