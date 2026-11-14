from typing import Callable, Optional, Sequence

import torch

from neupi.core.discretizer import BaseDiscretizer
from neupi.registry import register
from neupi.utils.pgm_utils import evaluate_log_likelihood
from neupi.utils.timing import (
    call_discretizer,
    deadline_from,
    expired,
    min_deadline,
    validate_time_limit,
)


@register("discretizer")
class BestOfDiscretizer(BaseDiscretizer):
    """
    Runs several discretizers and keeps, for each query, the assignment with the highest score.

    With an `OAUAI` and a `KNearestDiscretizer` this is the "OAUAI | k-Nearest" method
    from the SINE paper:

    ```
    BestOfDiscretizer(pgm_evaluator, [OAUAI(pgm_evaluator, k=...), KNearestDiscretizer(pgm_evaluator, k=1024)])
    ```

    Assignments are scored by the PGM evaluator; for MMAP the unobserved variables are
    marginalized out. On ties, the earlier discretizer in the list wins.

    Args:
        pgm_evaluator (Callable): The PGM evaluator (e.g., ProbabilisticGraphicalModel) which acts
                                  as the scoring function.
        discretizers (Sequence[BaseDiscretizer]): The discretizers to combine.
        time_limit (float, optional): Time budget in seconds for one call (one batch of
                          queries), shared by the discretizers in order: each one gets the
                          time that is left, and those that do not get to run are skipped.
                          If every discretizer is skipped, the thresholded assignment is
                          returned. None means no limit.

    References
    ----------
    Arya, Shivvrat, Rahman, Tahrima, and Gogate, Vibhav Giridhar. SINE: Scalable MPE Inference for Probabilistic Graphical Models Using Advanced Neural Embeddings. Proceedings of the 28th International Conference on Artificial Intelligence and Statistics (AISTATS), 2025.
    """

    def __init__(
        self,
        pgm_evaluator: Callable,
        discretizers: Sequence[BaseDiscretizer],
        time_limit: Optional[float] = None,
    ):
        super().__init__()
        if len(discretizers) == 0:
            raise ValueError("At least one discretizer is required.")
        self.pgm_evaluator = pgm_evaluator
        self.discretizers = list(discretizers)
        self.time_limit = validate_time_limit(time_limit)

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
        deadline = min_deadline(deadline, deadline_from(self.time_limit))
        # The incumbent if no discretizer gets to run
        best_assignments = (prob_outputs >= 0.5).to(prob_outputs.dtype)
        best_scores = None
        for discretizer in self.discretizers:
            if expired(deadline):
                break
            assignments = call_discretizer(
                discretizer,
                prob_outputs,
                query_mask,
                evidence_mask,
                unobs_mask,
                deadline=deadline,
                logits=logits,
            )
            scores = evaluate_log_likelihood(self.pgm_evaluator, assignments, unobs_mask)
            if best_scores is None:
                best_assignments, best_scores = assignments, scores
                continue
            better = scores > best_scores
            best_assignments = torch.where(better.unsqueeze(1), assignments, best_assignments)
            best_scores = torch.where(better, scores, best_scores)
        return best_assignments
