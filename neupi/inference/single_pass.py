from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from neupi.core.inference_module import BaseInferenceModule
from neupi.registry import register
from neupi.utils.pgm_utils import apply_evidence


@register("inference_module")
class SinglePassInferenceEngine(BaseInferenceModule):
    """
    Handles the inference process for a trained neural PGM solver.

    This class takes a trained model and runs it on a dataset, collecting
    the outputs. It operates in no_grad mode for efficiency.

    Args:
        model (torch.nn.Module): The trained neural network model.

    References
    ----------
    Arya, S., Rahman, T., & Gogate, V. (2025).
    SINE: Scalable MPE inference for probabilistic graphical models using advanced neural embeddings.
    AISTATS 2025. https://openreview.net/forum?id=6tZxEVlpnL

    Arya, S., Rahman, T., & Gogate, V. (2024).
    Learning to Solve the Constrained Most Probable Explanation Task in Probabilistic Graphical Models.
    arXiv preprint arXiv:2404.11606. https://arxiv.org/abs/2404.11606

    Arya, S., Rahman, T., & Gogate, V. (2024).
    Neural Network Approximators for Marginal MAP in Probabilistic Circuits.
    Proceedings of the AAAI Conference on Artificial Intelligence, 38(10), 10918–10926.
    https://doi.org/10.1609/aaai.v38i10.28966
    """

    def __init__(self, model: torch.nn.Module, discretizer: torch.nn.Module, device: str = "cpu"):
        self.model = model.to(device)
        self.discretizer = discretizer
        self.model.eval()  # Set model to evaluation mode
        self.device = device

    @torch.no_grad()
    def run(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass on the entire dataset to get predictions.

        Args:
            dataloader (DataLoader): DataLoader providing the inference data.
                Each batch should yield (inputs, evidence_data, evidence_mask).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing concatenated tensors for:
                - 'raw_outputs': Raw logits from the network.
                - 'prob_outputs': Probabilities after sigmoid activation.
                - 'final_assignments': Final assignments after applying evidence.
        """
        all_raw_outputs: List[torch.Tensor] = []
        all_prob_outputs: List[torch.Tensor] = []
        all_final_assignments: List[torch.Tensor] = []

        for batch_data in dataloader:
            evidence_data, evidence_mask, query_mask, unobs_mask = batch_data
            evidence_data = evidence_data.to(self.device)
            evidence_mask = evidence_mask.to(self.device)
            query_mask = query_mask.to(self.device)
            unobs_mask = unobs_mask.to(self.device)

            raw_predictions = self.model(evidence_data, evidence_mask, query_mask, unobs_mask)
            prob_predictions = torch.sigmoid(raw_predictions)
            final_assignments = apply_evidence(prob_predictions, evidence_data, evidence_mask)
            # Apply the discretizer to the final assignments
            final_assignments = self.discretizer(final_assignments)
            all_raw_outputs.append(raw_predictions.cpu())
            all_prob_outputs.append(prob_predictions.cpu())
            all_final_assignments.append(final_assignments.cpu())

        # Concatenate all batch results into single tensors
        results = {
            "raw_outputs": torch.cat(all_raw_outputs, dim=0),
            "prob_outputs": torch.cat(all_prob_outputs, dim=0),
            "final_assignments": torch.cat(all_final_assignments, dim=0).int(),
        }
        return results
