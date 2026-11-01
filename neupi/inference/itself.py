import copy
from typing import Dict, List, Optional, Type

import torch
from torch.utils.data import DataLoader

from neupi.core.inference_module import BaseInferenceModule
from neupi.losses import entropy_penalty
from neupi.registry import register
from neupi.utils.pgm_utils import apply_evidence, check_marginalization, mark_unobserved


@register("inference_module")
class ITSELF_Engine(BaseInferenceModule):
    """
    Handles Inference Time Self-Supervised Training (ITSELF).

    For each batch of data, this engine performs a few steps of fine-tuning
    on a copy of the model to refine the predictions for that specific batch.

    Args:
        model (torch.nn.Module): The base, pre-trained neural network model.
        pgm_evaluator (torch.nn.Module): The PGM evaluator for loss calculation.
        loss_fn (callable): The loss function used during refinement.
        optimizer_cls (Type[torch.optim.Optimizer]): The class of the optimizer to use
                                                     for refinement (e.g., torch.optim.Adam).
        refinement_lr (float): The learning rate for the test-time optimizer.
        refinement_steps (int): The number of optimization steps to perform per instance.
                                With `tol` set, this is the maximum number of steps.
        device (str): The device to run inference on ('cpu' or 'cuda').
        tol (float, optional): Stop refining a batch early once the loss has changed by
                               less than `tol` (absolute) for `patience` consecutive steps.
                               If None (default), all `refinement_steps` steps are run.
        patience (int): Number of consecutive steps below `tol` needed to stop. Defaults to 1.
        entropy_weight (float): Weight alpha of the entropy penalty over the query variables
                               (`neupi.losses.entropy_penalty`) added to the refinement loss.
                               Defaults to 0.0 (no penalty).
        entropy_aggregate (str): "avg" or "sum" over the batch for the entropy penalty.
                               Set it to match the loss's `aggregate`. Defaults to "avg".
        freeze_batchnorm (bool, optional): Whether BatchNorm layers use their running
                               statistics (eval mode) during refinement, with their affine
                               parameters still trained. If None (default), they are frozen
                               only for batches of a single query, where train-mode BatchNorm
                               cannot run; this allows per-query ITSELF.
    References
    ----------
    Arya, S., Rahman, T., & Gogate, V. G. (2024).
    A neural network approach for efficiently answering most probable explanation queries in probabilistic models.
    NeurIPS 2024. https://openreview.net/forum?id=ufPPf9ghzP
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pgm_evaluator: torch.nn.Module,
        loss_fn: callable,
        optimizer_cls: Type[torch.optim.Optimizer],
        discretizer: torch.nn.Module,
        refinement_lr: float,
        refinement_steps: int,
        device: str = "cpu",
        tol: Optional[float] = None,
        patience: int = 1,
        freeze_batchnorm: Optional[bool] = None,
        entropy_weight: float = 0.0,
        entropy_aggregate: str = "avg",
    ):
        if entropy_weight < 0:
            raise ValueError("entropy_weight must be non-negative.")
        if entropy_aggregate not in ("avg", "sum"):
            raise ValueError("entropy_aggregate must be 'avg' or 'sum'.")
        if tol is not None and tol < 0:
            raise ValueError("tol must be non-negative.")
        if patience < 1:
            raise ValueError("patience must be at least 1.")
        self.base_model = model.to(device)
        self.base_model.eval()

        self.pgm_evaluator = pgm_evaluator.to(device)
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls
        self.discretizer = discretizer
        self.refinement_lr = refinement_lr
        self.refinement_steps = refinement_steps
        self.device = device
        self.tol = tol
        self.patience = patience
        self.freeze_batchnorm = freeze_batchnorm
        self.entropy_weight = entropy_weight
        self.entropy_aggregate = entropy_aggregate

    # The @torch.no_grad() decorator has been removed from the run method.
    def run(self, data) -> Dict[str, torch.Tensor]:
        """
        Performs test-time refinement for each batch in the dataloader or a single batch.
        Args:
            data: Either a DataLoader or a tuple of (evidence_data, evidence_mask, query_mask, unobs_mask).
        Returns:
            Dict[str, torch.Tensor]: Final assignments.
        """
        all_final_assignments: List[torch.Tensor] = []

        # Determine if input is DataLoader or a single batch
        if isinstance(data, DataLoader):
            batch_iter = data
        elif isinstance(data, (tuple, list)) and len(data) == 4:
            batch_iter = [data]
        else:
            raise ValueError("Input must be a DataLoader or a tuple of four tensors.")

        for batch_data in batch_iter:
            final_assignment, _ = self._process_batch(batch_data)
            all_final_assignments.append(final_assignment.cpu())

        results = {"final_assignments": torch.cat(all_final_assignments, dim=0).int()}
        return results

    def _process_batch(self, batch_data, trained_model=None):
        """
        Process a single batch of data.
        Args:
            batch_data: A tuple of (evidence_data, evidence_mask, query_mask, unobs_mask).
            trained_model: A model that has been trained on a subset of the data.
        Returns:
            final_assignment: The final assignment for the batch.
            temp_model: The model that has been trained on the batch.
        """
        evidence_data, evidence_mask, query_mask, unobs_mask = batch_data
        if evidence_data.device != self.device:
            evidence_data = evidence_data.to(self.device)
            evidence_mask = evidence_mask.to(self.device)
            query_mask = query_mask.to(self.device)
            unobs_mask = unobs_mask.to(self.device)

        if trained_model is None:
            temp_model = copy.deepcopy(self.base_model)
        else:
            temp_model = trained_model

        # For MMAP, the unobserved variables are marginalized out of the loss
        marginalize = bool(unobs_mask.any())
        if marginalize:
            check_marginalization(self.pgm_evaluator)

        temp_model.train()
        freeze_batchnorm = self.freeze_batchnorm
        if freeze_batchnorm is None:
            freeze_batchnorm = evidence_data.shape[0] == 1
        if freeze_batchnorm:
            for module in temp_model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        optimizer = self.optimizer_cls(temp_model.parameters(), lr=self.refinement_lr)

        prev_loss = None
        steps_below_tol = 0
        for _ in range(self.refinement_steps):
            optimizer.zero_grad()
            raw_preds = temp_model(evidence_data, evidence_mask, query_mask, unobs_mask)
            prob_preds = torch.sigmoid(raw_preds)
            final_assigns = apply_evidence(prob_preds, evidence_data, evidence_mask)
            if marginalize:
                loss = self.loss_fn(mark_unobserved(final_assigns, unobs_mask), self.pgm_evaluator)
            else:
                loss = self.loss_fn(final_assigns, self.pgm_evaluator)
            if self.entropy_weight:
                loss = loss + entropy_penalty(
                    final_assigns, query_mask, self.entropy_weight, self.entropy_aggregate
                )
            loss.backward()
            optimizer.step()

            # Stop once the loss has converged
            if self.tol is not None:
                loss_value = loss.item()
                if prev_loss is not None and abs(prev_loss - loss_value) < self.tol:
                    steps_below_tol += 1
                    if steps_below_tol >= self.patience:
                        break
                else:
                    steps_below_tol = 0
                prev_loss = loss_value

        with torch.no_grad():
            temp_model.eval()
            final_raw = temp_model(evidence_data, evidence_mask, query_mask, unobs_mask)
            final_prob = torch.sigmoid(final_raw)
            final_assignment = apply_evidence(final_prob, evidence_data, evidence_mask)
            final_assignment = self.discretizer(
                final_assignment, query_mask, evidence_mask, unobs_mask
            )
        return final_assignment, temp_model
