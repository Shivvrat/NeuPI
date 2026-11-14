import time
from typing import Dict, Iterable, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from neupi.core.inference_module import BaseInferenceModule
from neupi.registry import register
from neupi.utils.pgm_utils import apply_evidence
from neupi.utils.timing import (
    call_discretizer,
    deadline_from,
    expired,
    min_deadline,
    stage_time_limit,
    validate_time_limit,
)


@register("inference_module")
class SinglePassInferenceEngine(BaseInferenceModule):
    """
    Handles the inference process for a trained neural PGM solver.

    This class takes a trained model and runs it on a dataset, collecting
    the outputs. It operates in no_grad mode for efficiency.

    Args:
        model (torch.nn.Module): The trained neural network model.
        discretizer (torch.nn.Module): Turns the probabilities into binary assignments.
        device (str): The device to run inference on ('cpu' or 'cuda').
        time_limit (float, optional): Time budget in seconds for answering one batch of
            queries (use a batch size of 1 for a per-query budget): the forward pass plus the
            discretizer. The forward pass always completes; the discretizer gets the time
            that is left and returns its best incumbent when it runs out (at worst the
            thresholded assignment). None means no limit.
        discretizer_time_limit (float or str, optional): Time budget for the discretizer
            alone, in addition to `time_limit`: a number of seconds, or a percentage of
            `time_limit` such as ``"50%"``.

    `run` also returns, per batch, ``elapsed_seconds`` and ``time_limit_reached``.

    References
    ----------
    Arya, S., Rahman, T., & Gogate, V. (2025).
    SINE: Scalable MPE inference for probabilistic graphical models using advanced neural embeddings.
    AISTATS 2025. https://openreview.net/forum?id=6tZxEVlpnL

    Arya, S., Rahman, T., & Gogate, V. (2024). Learning to Solve the Constrained Most Probable Explanation Task in Probabilistic Graphical Models. Proceedings of The 27th International Conference on Artificial Intelligence and Statistics. International Conference on Artificial Intelligence and Statistics, PMLR, pp. 2791–2799. https://proceedings.mlr.press/v238/arya24b.html

    Arya, S., Rahman, T., & Gogate, V. (2024).
    Neural Network Approximators for Marginal MAP in Probabilistic Circuits.
    Proceedings of the AAAI Conference on Artificial Intelligence, 38(10), 10918–10926.
    https://doi.org/10.1609/aaai.v38i10.28966
    """

    def __init__(
        self,
        model: torch.nn.Module,
        discretizer: torch.nn.Module,
        device: str = "cpu",
        time_limit: Optional[float] = None,
        discretizer_time_limit: Optional[Union[float, str]] = None,
    ):
        self.model = model.to(device)
        self.discretizer = discretizer
        self.model.eval()  # Set model to evaluation mode
        self.device = device
        self.time_limit = validate_time_limit(time_limit)
        self.discretizer_time_limit = stage_time_limit(
            discretizer_time_limit, self.time_limit, "discretizer_time_limit"
        )

    @torch.no_grad()
    def run(self, data) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass on the entire dataset or a single batch to get predictions.

        Args:
            data: Either a DataLoader or a tuple/list of (evidence_data, evidence_mask, query_mask, unobs_mask).

        Returns:
            Dict[str, torch.Tensor]: Concatenated tensors for:
                - 'raw_outputs': Raw logits from the network.
                - 'prob_outputs': Probabilities after sigmoid activation.
                - 'final_assignments': Final assignments after applying evidence.
                - 'elapsed_seconds': Time taken per batch, shape (num_batches,).
                - 'time_limit_reached': Per batch, whether a time limit cut the search short.
        """
        all_raw_outputs: List[torch.Tensor] = []
        all_prob_outputs: List[torch.Tensor] = []
        all_final_assignments: List[torch.Tensor] = []
        elapsed: List[float] = []
        limit_reached: List[bool] = []

        # Determine input type
        batch_iter: Iterable
        if isinstance(data, DataLoader):
            batch_iter = data
        elif isinstance(data, (tuple, list)) and len(data) == 4:
            batch_iter = [data]
        else:
            raise ValueError("Input must be a DataLoader or a tuple/list of four tensors.")

        for batch_data in batch_iter:
            raw, prob, final, timing = self._process_batch(batch_data)
            all_raw_outputs.append(raw.cpu())
            all_prob_outputs.append(prob.cpu())
            all_final_assignments.append(final.cpu())
            elapsed.append(timing["elapsed_seconds"])
            limit_reached.append(timing["time_limit_reached"])

        results = {
            "raw_outputs": torch.cat(all_raw_outputs, dim=0),
            "prob_outputs": torch.cat(all_prob_outputs, dim=0),
            "final_assignments": torch.cat(all_final_assignments, dim=0).int(),
            "elapsed_seconds": torch.tensor(elapsed, dtype=torch.float64),
            "time_limit_reached": torch.tensor(limit_reached, dtype=torch.bool),
        }
        return results

    def _process_batch(self, batch_data):
        start = time.monotonic()
        total_deadline = deadline_from(self.time_limit, start)
        evidence_data, evidence_mask, query_mask, unobs_mask = batch_data
        evidence_data = evidence_data.to(self.device)
        evidence_mask = evidence_mask.to(self.device)
        query_mask = query_mask.to(self.device)
        unobs_mask = unobs_mask.to(self.device)

        raw_predictions = self.model(evidence_data, evidence_mask, query_mask, unobs_mask)
        prob_predictions = torch.sigmoid(raw_predictions)
        final_assignments = apply_evidence(prob_predictions, evidence_data, evidence_mask)
        # The discretizer gets whatever is left of the total budget, capped by its own budget
        discretizer_deadline = min_deadline(
            total_deadline, deadline_from(self.discretizer_time_limit)
        )
        final_assignments = call_discretizer(
            self.discretizer,
            final_assignments,
            query_mask,
            evidence_mask,
            unobs_mask,
            deadline=discretizer_deadline,
            logits=raw_predictions,
        )
        timing = {
            "elapsed_seconds": time.monotonic() - start,
            "time_limit_reached": expired(discretizer_deadline),
        }
        return raw_predictions, prob_predictions, final_assignments, timing
