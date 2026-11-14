import copy
import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from neupi.core.trainer import BaseTrainer
from neupi.losses import entropy_penalty
from neupi.registry import register
from neupi.utils.pgm_utils import (
    apply_evidence,
    check_marginalization,
    evaluate_log_likelihood,
    mark_unobserved,
)
from torch.utils.data import DataLoader


@register("trainer")
class SelfSupervisedTrainer(BaseTrainer):
    """
    A trainer for self-supervised learning of neural PGM solvers.

    This class handles the training and validation loops, including the
    forward pass, loss calculation, backpropagation, and optimizer steps.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        pgm_evaluator (torch.nn.Module): The PGM evaluator (e.g., ProbabilisticGraphicalModel)
                                         used for calculating log-likelihoods.
        loss_fn (callable): The loss function. It should accept predictions
                            and the PGM evaluator as arguments.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        device (str): The device to run training on ('cpu' or 'cuda').
        entropy_weight (float): Weight alpha of the entropy penalty over the query
                                variables (`neupi.losses.entropy_penalty`) added to the loss.
                                Negative values give an entropy bonus. Defaults to 0.0 (no
                                penalty).
        entropy_aggregate (str): "avg" or "sum" over the batch for the entropy penalty.
                                 Set it to match the loss's `aggregate`. Defaults to "avg".
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pgm_evaluator: torch.nn.Module,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        entropy_weight: float = 0.0,
        entropy_aggregate: str = "avg",
    ):
        if entropy_aggregate not in ("avg", "sum"):
            raise ValueError("entropy_aggregate must be 'avg' or 'sum'.")
        self.model = model.to(device)
        self.pgm_evaluator = pgm_evaluator.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.entropy_weight = entropy_weight
        self.entropy_aggregate = entropy_aggregate

    def _compute_loss(self, batch_data):
        """Computes the loss of the model's predictions on a batch of data."""
        # Unpack batch data (assuming a tuple of tensors)
        # You might need to adjust this based on your DataLoader's output
        evidence_data, evidence_mask, query_mask, unobs_mask = batch_data
        evidence_data = evidence_data.to(self.device)
        evidence_mask = evidence_mask.to(self.device)
        query_mask = query_mask.to(self.device)
        unobs_mask = unobs_mask.to(self.device)

        # Forward pass
        raw_predictions = self.model(evidence_data, evidence_mask, query_mask, unobs_mask)

        # Apply a sigmoid to get probabilities in the range [0, 1]
        predictions = torch.sigmoid(raw_predictions)

        # Process predictions: apply evidence
        # The network's predictions for evidence variables are replaced with their true values.
        final_assignments = apply_evidence(predictions, evidence_data, evidence_mask)

        # Calculate loss; for MMAP, the unobserved variables are marginalized out
        if unobs_mask.any():
            check_marginalization(self.pgm_evaluator)
            loss = self.loss_fn(mark_unobserved(final_assignments, unobs_mask), self.pgm_evaluator)
        else:
            loss = self.loss_fn(final_assignments, self.pgm_evaluator)
        if self.entropy_weight:
            loss = loss + entropy_penalty(
                final_assignments, query_mask, self.entropy_weight, self.entropy_aggregate
            )
        return loss

    def step(self, batch_data):
        """Performs a single training step on a batch of data."""
        self.model.train()
        # BatchNorm cannot compute batch statistics from one sample (e.g. the last batch of
        # an epoch), so for such a batch it uses its running statistics instead
        if batch_data[0].shape[0] == 1:
            for module in self.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        loss = self._compute_loss(batch_data)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> float:
        """Returns the average loss of the model on a DataLoader, in eval mode."""
        self.model.eval()
        total_loss = 0
        for batch_data in dataloader:
            total_loss += self._compute_loss(batch_data).item()
        return total_loss / len(dataloader)

    @torch.no_grad()
    def evaluate_assignments(self, dataloader: DataLoader) -> float:
        """
        Returns the mean negative log-likelihood of the model's thresholded assignments on a
        DataLoader (query variables set to 1 where the logit is positive, evidence kept), in
        eval mode. For MMAP the unobserved variables are marginalized out.
        """
        self.model.eval()
        total, count = 0.0, 0
        for batch_data in dataloader:
            evidence_data, evidence_mask, query_mask, unobs_mask = (
                t.to(self.device) for t in batch_data
            )
            logits = self.model(evidence_data, evidence_mask, query_mask, unobs_mask)
            assignments = apply_evidence((logits > 0).to(logits.dtype), evidence_data, evidence_mask)
            ll = evaluate_log_likelihood(self.pgm_evaluator, assignments, unobs_mask)
            total += -ll.double().sum().item()
            count += ll.numel()
        return total / count

    def _selection_score(self, selection, val_dataloader: DataLoader, val_loss: float) -> float:
        if selection == "loss":
            return val_loss
        if selection == "assignment":
            return self.evaluate_assignments(val_dataloader)
        self.model.eval()
        with torch.no_grad():
            return float(selection(self.model, val_dataloader))

    def fit(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        val_dataloader: Optional[DataLoader] = None,
        scheduler=None,
        verbose: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], Optional[bool]]] = None,
        selection: Union[str, Callable[[torch.nn.Module, DataLoader], float]] = "assignment",
    ):
        """
        Runs the full training loop for a specified number of epochs.

        Args:
            dataloader (DataLoader): The DataLoader providing training data.
            num_epochs (int): The number of epochs to train for.
            val_dataloader (DataLoader, optional): If given, the loss and the `selection`
                score are computed on it after every epoch, and the weights from the epoch
                with the lowest score are restored at the end (the epoch is stored in
                `best_epoch`, its score in `best_val_score` and its loss in
                `best_val_loss`). If None, the final epoch's weights are kept.
            scheduler (optional): A learning-rate scheduler stepped once per epoch.
                `torch.optim.lr_scheduler.ReduceLROnPlateau` is stepped with the
                validation loss (training loss if there is no validation set), e.g.
                `ReduceLROnPlateau(optimizer, factor=0.9)`; other schedulers are stepped
                without arguments.
            verbose (bool): Print the losses after every epoch. Defaults to True.
            callback (callable, optional): Called after every epoch with a dict with keys
                "epoch" (1-based), "num_epochs", "train_loss", "val_loss" (None without a
                validation set), "lr" and "seconds" (the epoch's duration). Returning True
                stops training after that epoch, e.g. for early stopping or a time limit.
                With a validation set the dict also has "val_score".
            selection (str or callable): How the epoch to keep is chosen on the validation
                set (lower is better):

                - "assignment" (default): the mean negative log-likelihood of the
                  thresholded assignments (`evaluate_assignments`), i.e. the quality of
                  the answers the network gives.
                - "loss": the validation loss (relaxed predictions, with the entropy term).
                - a function `selection(model, val_dataloader) -> float`, called with the
                  model in eval mode and without gradients, e.g. to score assignments from
                  another discretizer.

        The per-epoch dicts are also kept in `self.history`.
        """
        if not callable(selection) and selection not in ("assignment", "loss"):
            raise ValueError("selection must be 'assignment', 'loss' or a callable.")
        best_state = None
        self.best_val_loss = float("inf")
        self.best_val_score = float("inf")
        self.best_epoch: Optional[int] = None
        self.history: List[Dict[str, Any]] = []
        for epoch in range(num_epochs):
            start = time.monotonic()
            total_loss = 0.0
            for batch_data in dataloader:
                loss = self.step(batch_data)
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            message = f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}"

            monitored_loss = avg_loss
            val_loss = None
            val_score = None
            if val_dataloader is not None:
                val_loss = self.evaluate(val_dataloader)
                message += f", Validation Loss: {val_loss:.4f}"
                monitored_loss = val_loss
                val_score = self._selection_score(selection, val_dataloader, val_loss)
                if selection != "loss":
                    message += f", Validation Score: {val_score:.4f}"
                if val_score < self.best_val_score:
                    self.best_val_score = val_score
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    best_state = copy.deepcopy(self.model.state_dict())
            if verbose:
                print(message)

            info = {
                "epoch": epoch + 1,
                "num_epochs": num_epochs,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
                "seconds": time.monotonic() - start,
            }
            self.history.append(info)

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitored_loss)
                else:
                    scheduler.step()

            if val_score is not None:
                info["val_score"] = val_score

            if callback is not None and callback(info):
                if verbose:
                    print(f"Stopped by the callback after epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(
                    f"Restored weights from epoch {self.best_epoch} "
                    f"(Validation Score: {self.best_val_score:.4f}, Loss: {self.best_val_loss:.4f})"
                )

        return self.model
