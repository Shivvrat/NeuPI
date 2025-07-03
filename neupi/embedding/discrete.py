import torch

from neupi.core.embedding import Embedding
from neupi.registry import register


@register("embedding")
class DiscreteEmbedder(Embedding):
    """
    Creates feature embeddings from variable assignments and bucket information.

    This preprocessor converts a tensor of binary variable assignments into a
    new feature space suitable for input to a neural network. It expands each
    variable `v` into a two-dimensional representation `[v, 1-v]` and then
    replaces the features for query and unobserved variables with specific
    embedding values.

    Args:
        query_embedding (float): The value to use for query variables. Defaults to 0.0.
        unobserved_embedding (float): The value to use for unobserved variables. Defaults to 1.0.
    """

    def __init__(
        self, num_vars: int, query_embedding: float = 0.0, unobserved_embedding: float = 1.0
    ):
        self.num_vars = num_vars
        self.embedding_size = num_vars * 2
        self.query_embedding = query_embedding
        self.unobserved_embedding = unobserved_embedding

    def __call__(
        self,
        evidence_data: torch.Tensor,
        evidence_mask: torch.Tensor,
        query_mask: torch.Tensor,
        unobs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Processes the assignments to create embeddings.

        Args:
            assignments (torch.Tensor): A tensor of binary assignments.
                Shape: (num_vars,) or (batch_size, num_vars).
            buckets (Dict[str, torch.Tensor]): A dictionary containing boolean masks for different
                variable types. Expected keys: 'query', 'unobs'.

        Returns:
            torch.Tensor: The embedded feature tensor.
                Shape: (num_vars*2,) or (batch_size, num_vars*2).
        """
        is_batch = evidence_data.dim() == 2
        if not is_batch:
            # Add a temporary batch dimension for consistent processing
            evidence_data = evidence_data.unsqueeze(0)

        num_samples, n_vars = evidence_data.size()
        device = evidence_data.device
        dtype = evidence_data.dtype

        # Create the expanded feature tensor [v, 1-v]
        embedded_features = torch.zeros(num_samples, n_vars * 2, dtype=dtype, device=device)
        embedded_features[:, 0::2] = evidence_data  # Even indices get the original value
        embedded_features[:, 1::2] = 1 - evidence_data  # Odd indices get 1 - value

        # Apply embeddings for query and unobserved variables
        if query_mask is not None:
            embedded_features[:, 0::2][query_mask] = self.query_embedding
            embedded_features[:, 1::2][query_mask] = self.query_embedding

        if unobs_mask is not None:
            embedded_features[:, 0::2][unobs_mask] = self.unobserved_embedding
            embedded_features[:, 1::2][unobs_mask] = self.unobserved_embedding

        if not is_batch:
            # Remove the temporary batch dimension if the input was a single instance
            return embedded_features.squeeze(0)

        return embedded_features
