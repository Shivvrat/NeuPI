# neupi/core/trainer.py
import abc

from torch import nn


class BaseTrainer(abc.ABC):
    """Abstract base class for all training strategies."""

    def __init__(self, model: nn.Module, prob_model: nn.Module, **kwargs):
        self.model = model
        self.prob_model = prob_model
        # Initialize optimizer, etc.
        ...

    def step(self, batch):
        """Performs a single optimization step."""
        ...

    def fit(self, dataloader, *args, **kwargs):
        """Runs the main training loop."""
        ...
