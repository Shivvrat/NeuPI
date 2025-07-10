import abc

import torch


class BaseDiscretizer(torch.nn.Module):
    """Abstract base class for all discretizers."""

    @abc.abstractmethod
    def __call__(
        self,
        prob_outputs: torch.Tensor,
        query_mask: torch.Tensor = None,
        evidence_mask: torch.Tensor = None,
        unobs_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        raise NotImplementedError
