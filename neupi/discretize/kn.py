from typing import Callable, Optional, Union

import numpy as np
import torch

from neupi.core.discretizer import BaseDiscretizer, resolve_k, validate_k
from neupi.registry import register
from neupi.utils.pgm_utils import evaluate_log_likelihood
from neupi.utils.timing import deadline_from, expired, min_deadline, validate_time_limit

from .cython_kn.kn_binary_vectors import cython_process_assignments


def k_nearest_from_logits(logits: np.ndarray, k: int) -> np.ndarray:
    """
    The k binary vectors nearest to sigmoid(logits) in L1 distance (Algorithm 1 of SINE),
    computed without rounding the probabilities.

    D(q, b) = sum_i |q_i - b_i| is a constant plus |2 q_i - 1| = 1 - 2 e_i for every variable
    where b_i differs from the thresholded value, with e_i = sigmoid(-|logit_i|). The beam
    search of Algorithm 1 runs over these flips with the cost of a set of flips kept as
    (number of flips - 2 * sum of e_i), ties in float64 broken by the larger sum of e_i, so
    that near-saturated probabilities (which round to exactly 0 or 1) keep their order.

    Returns an array of shape (min(k, 2^n), n) of 0/1 values, nearest first.
    """
    logits = np.asarray(logits, dtype=np.float64)
    n = logits.shape[0]
    base = logits >= 0
    e = 1.0 / (
        1.0 + np.exp(np.abs(logits))
    )  # sigmoid(-|logit|), the distance to the nearer of 0 and 1
    count = np.zeros(1, dtype=np.int64)
    total_e = np.zeros(1)
    parents = []
    flips = []
    for i in range(n):
        # Every partial assignment either keeps variable i at its thresholded value or flips it
        count_new = np.concatenate([count, count + 1])
        e_new = np.concatenate([total_e, total_e + e[i]])
        order = np.lexsort((-e_new, count_new - 2.0 * e_new))[:k]
        parents.append(order % len(count))
        flips.append(order >= len(count))
        count, total_e = count_new[order], e_new[order]
    # Walk back through the beams to recover the flipped variables of each final assignment
    candidates = np.tile(base, (len(count), 1))
    index = np.arange(len(count))
    for i in range(n - 1, -1, -1):
        candidates[:, i] ^= flips[i][index]
        index = parents[i][index]
    return candidates.astype(np.int8)


@register("discretizer_knn")
class KNearestDiscretizer(BaseDiscretizer):
    """
    Finds k-nearest binary vectors for query variables using a scoring function.

    This method generates candidate binary assignments for query variables that are
    "close" to the continuous predictions, scores them using the PGM evaluator,
    and selects the best one.

    Args:
        pgm_evaluator (Callable): The PGM evaluator (e.g., ProbabilisticGraphicalModel) which acts
                                  as the scoring function.
        k (int or float): Number of nearest binary vectors to consider. A float in
                          (0, 1) is a fraction of each query's query variables
                          (ceil(k * num_query_vars), at least 1), so k varies per query.
        batch_size (int): Batch size for scoring candidate assignments.
        max_k (int, optional): Upper limit on the per-query k.
        time_limit (float, optional): Time budget in seconds for one call (one batch of
                          queries). When it runs out, the queries not yet searched keep
                          their thresholded assignment and the query being searched keeps
                          the best candidate scored so far. None means no limit.


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
        time_limit: Optional[float] = None,
    ):
        super().__init__()
        validate_k(k, max_k)
        self.time_limit = validate_time_limit(time_limit)
        self.max_k = max_k
        self.pgm_evaluator = pgm_evaluator
        self.k = k
        self.batch_size = batch_size

    @torch.no_grad()
    def __call__(
        self,
        prob_outputs: torch.Tensor,
        query_mask: torch.Tensor,
        evidence_mask: Optional[torch.Tensor] = None,
        unobs_mask: Optional[torch.Tensor] = None,
        deadline: Optional[float] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Selects k-nearest binary vectors for each continuous output from the NN and selects the best one.

        Args:
            prob_outputs (torch.Tensor): Continuous predictions from the network.
                Shape: (batch_size, num_variables).
            evidence_mask (torch.Tensor): Boolean mask for evidence variables.
            query_mask (torch.Tensor): Boolean mask for query variables.
            unobs_mask (torch.Tensor): Boolean mask for unobserved variables.
            deadline (float, optional): Absolute `time.monotonic()` deadline from the caller;
                combined with this discretizer's own `time_limit`.
            logits (torch.Tensor, optional): The network outputs before the sigmoid (passed
                by the inference engines). If given, the distances are computed from them
                (`k_nearest_from_logits`), so probabilities that have saturated to exactly 0
                or 1 in floating point are still ordered correctly.

        Returns:
            torch.Tensor: The final discrete assignments.
                Shape: (batch_size, num_variables).
        """
        num_examples, num_vars = prob_outputs.shape
        device = prob_outputs.device
        dtype = prob_outputs.dtype

        deadline = min_deadline(deadline, deadline_from(self.time_limit))
        query_indices = [torch.where(qm)[0] for qm in query_mask]
        thresholded_assignments = (prob_outputs >= 0.5).to(dtype)
        best_assignments = thresholded_assignments.clone()

        for i in range(num_examples):
            # Out of time: the remaining queries keep their thresholded assignment
            if expired(deadline):
                break
            # Nothing to search for a query without query variables
            if len(query_indices[i]) == 0:
                continue
            k = resolve_k(self.k, len(query_indices[i]), self.max_k)
            if logits is not None:
                candidate_np = k_nearest_from_logits(
                    logits[i, query_indices[i]].detach().cpu().numpy(), k
                )
            else:
                # The Cython helper takes a float32 buffer
                query_probs_np = (
                    prob_outputs[i, query_indices[i]]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                # Get k candidate binary assignments from the Cython helper
                _, candidate_np = cython_process_assignments(query_probs_np, k)
            candidates = torch.tensor(candidate_np, dtype=dtype, device=device)

            best_score = torch.tensor(float("-inf"), device=device)
            best_local_assignment = None

            # Score each candidate assignment in batches
            for j in range(0, candidates.shape[0], self.batch_size):
                if j > 0 and expired(deadline):
                    break
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

            if best_local_assignment is not None:
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
