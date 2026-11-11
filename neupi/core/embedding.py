import torch


class Embedding:
    """
    Base class for all embeddings.

    Subclasses set `embedding_size`, the number of features they produce per query.
    """

    embedding_size: int

    def __call__(
        self,
        evidence_data: torch.Tensor,
        evidence_mask: torch.Tensor,
        query_mask: torch.Tensor,
        unobs_mask: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError
