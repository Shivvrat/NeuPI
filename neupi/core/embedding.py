import torch


class Embedding:
    """
    Base class for all embeddings.
    """

    def __call__(
        self,
        evidence_data: torch.Tensor,
        evidence_mask: torch.Tensor,
        query_mask: torch.Tensor,
        unobs_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
