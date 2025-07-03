import torch
from torch.utils.data import DataLoader

from neupi.core.trainer import BaseTrainer
from neupi.registry import register
from neupi.utils.pgm_utils import apply_evidence


@register("trainer")
class SelfSupervisedTrainer(BaseTrainer):
    """
    A trainer for self-supervised learning of neural PGM solvers.

    This class handles the training and validation loops, including the
    forward pass, loss calculation, backpropagation, and optimizer steps.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        pgm_evaluator (torch.nn.Module): The PGM evaluator (e.g., MarkovNetwork)
                                         used for calculating log-likelihoods.
        loss_fn (callable): The loss function. It should accept predictions
                            and the PGM evaluator as arguments.
        optimizer (torch.optim.Optimizer): The optimizer for updating model weights.
        device (str): The device to run training on ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pgm_evaluator: torch.nn.Module,
        loss_fn: callable,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.pgm_evaluator = pgm_evaluator.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def step(self, batch_data):
        """Performs a single training step on a batch of data."""
        # Unpack batch data (assuming a tuple of tensors)
        # You might need to adjust this based on your DataLoader's output
        inputs, evidence_data, evidence_mask = batch_data
        inputs = inputs.to(self.device)
        evidence_data = evidence_data.to(self.device)
        evidence_mask = evidence_mask.to(self.device)

        # Forward pass
        self.model.train()
        raw_predictions = self.model(inputs)

        # Apply a sigmoid to get probabilities in the range [0, 1]
        predictions = torch.sigmoid(raw_predictions)

        # Process predictions: apply evidence
        # The network's predictions for evidence variables are replaced with their true values.
        final_assignments = apply_evidence(predictions, evidence_data, evidence_mask)

        # Calculate loss
        loss = self.loss_fn(final_assignments, self.pgm_evaluator)

        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def fit(self, dataloader: DataLoader, num_epochs: int):
        """
        Runs the full training loop for a specified number of epochs.

        Args:
            dataloader (DataLoader): The DataLoader providing training data.
            num_epochs (int): The number of epochs to train for.
        """
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_data in dataloader:
                loss = self.step(batch_data)
                total_loss += loss

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        return self.model

    # You can add a similar `validate` method here for a validation loop.
