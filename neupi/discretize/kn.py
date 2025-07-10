from typing import Callable

import torch

from neupi.core.discretizer import BaseDiscretizer
from neupi.registry import register

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
        k (int): Number of nearest binary vectors to consider.
        batch_size (int): Batch size for scoring candidate assignments.


    References
    ----------
    Arya, Shivvrat, Rahman, Tahrima, and Gogate, Vibhav Giridhar. SINE: Scalable MPE Inference for Probabilistic Graphical Models Using Advanced Neural Embeddings. Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS), 2025.
    """

    def __init__(self, pgm_evaluator: Callable, k: int, batch_size: int = 300):
        super().__init__()
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
        best_assignments = prob_outputs.clone()

        for i in range(num_examples):
            query_probs_np = prob_outputs[i, query_indices[i]].detach().cpu().numpy()

            # Get k candidate binary assignments from the Cython helper
            _, candidate_np = cython_process_assignments(query_probs_np, self.k)
            candidates = torch.tensor(candidate_np, dtype=dtype, device=device)

            best_score = torch.tensor(float("-inf"), device=device)
            best_local_assignment = None

            # Score each candidate assignment in batches
            for j in range(0, candidates.shape[0], self.batch_size):
                batch_candidates = candidates[j : j + self.batch_size]

                # Create a batch of full assignments, each with a different candidate
                data_batch = prob_outputs[i].unsqueeze(0).repeat(len(batch_candidates), 1)
                data_batch[:, query_indices[i]] = batch_candidates

                scores = self.pgm_evaluator(data_batch)
                max_score, max_idx = torch.max(scores, dim=0)

                if max_score > best_score:
                    best_score = max_score
                    best_local_assignment = batch_candidates[max_idx]

            best_assignments[i, query_indices[i]] = best_local_assignment

        # Compare with simple thresholding and return the better result
        thresholded_assignments = (prob_outputs >= 0.5).to(dtype)
        thresholded_scores = self.pgm_evaluator(thresholded_assignments)
        best_scores = self.pgm_evaluator(best_assignments)

        final_assignments = torch.where(
            best_scores.unsqueeze(1) > thresholded_scores.unsqueeze(1),
            best_assignments,
            thresholded_assignments,
        )
        return final_assignments
