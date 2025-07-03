import abc

import torch


class BaseDiscretizer(torch.nn.Module):
    """Abstract base class for all discretizers."""

    @torch.no_grad()
    @abc.abstractmethod
    def forward(self, prob_outputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
