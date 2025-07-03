import torch

from neupi.core.embedding import Embedding
from neupi.registry import register


@register("embedding")
class IdentityEmbedding(Embedding):
    """
    An identity embedding that does not change the input.
    """

    def __init__(self, num_vars: int):
        print("Initializing IdentityEmbedding")
        print(
            "This is not recommended; since it uses the complete assignment as input. Masking out query variables is recommended."
        )
        self.num_vars = num_vars
        self.embedding_size = num_vars

    def __call__(
        self,
        evidence_data: torch.Tensor,
        evidence_mask: torch.Tensor,
        query_mask: torch.Tensor,
        unobs_mask: torch.Tensor,
    ) -> torch.Tensor:
        return evidence_data
