from typing import List, Type

import torch
import torch.nn as nn

from neupi.core.embedding import Embedding
from neupi.core.model import BaseNNModel
from neupi.registry import register


@register("nn_model")
class MLP(BaseNNModel):
    """
    A flexible Multi-Layer Perceptron (MLP) network.

    This module creates a fully connected neural network with configurable
    hidden layers, activation functions, batch normalization, and dropout. We extract the input size from the embedding.

    Args:
        hidden_sizes (List[int]): A list where each element is the number of
                                  neurons in a hidden layer.
        output_size (int): The number of neurons in the output layer.
        embedding (Embedding): Turns the four query tensors into the network input and
                               sets the input size (`embedding.embedding_size`), e.g.
                               `DiscreteEmbedder(num_vars)` or `IdentityEmbedding(num_vars)`.
        hidden_activation (str): The activation function for hidden layers.
                                 Supported: 'relu', 'leaky_relu'.
        use_batchnorm (bool): If True, adds a BatchNorm1d layer after each hidden
                              linear layer, before the activation. Defaults to False,
                              as in the papers. Models saved with neupi <= 1.1.0
                              defaults need `use_batchnorm=True` to load.
        dropout_rate (float): The dropout probability. If 0.0, no dropout is
                              applied. Defaults to 0.0.
    """

    def __init__(
        self,
        hidden_sizes: List[int],
        output_size: int,
        embedding: Embedding,
        hidden_activation: str = "relu",
        use_batchnorm: bool = False,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        if embedding is None:
            raise ValueError(
                "MLP needs an embedding to size its input layer; use IdentityEmbedding(num_vars) "
                "to feed the raw assignment."
            )
        self.embedding = embedding
        # The constructor arguments, used by neupi.save_solver to rebuild the network
        self.config = {
            "hidden_sizes": list(hidden_sizes),
            "output_size": output_size,
            "hidden_activation": hidden_activation,
            "use_batchnorm": use_batchnorm,
            "dropout_rate": dropout_rate,
        }

        activation_fn: Type[nn.Module]
        if hidden_activation == "relu":
            activation_fn = nn.ReLU
        elif hidden_activation == "leaky_relu":
            activation_fn = nn.LeakyReLU
        else:
            raise ValueError("Unsupported activation function.")

        layers: List[nn.Module] = []
        current_size = embedding.embedding_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_fn())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            current_size = hidden_size

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(current_size, output_size)

        self.initialize_weights(hidden_activation)

    def get_config(self) -> dict:
        """The constructor arguments other than `embedding`."""
        return dict(self.config)

    def initialize_weights(self, nonlinearity: str = "relu"):
        """
        Initializes the weights of the network using appropriate methods.
        """
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_in",
                    nonlinearity=nonlinearity,  # type: ignore[arg-type]
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        evidence_data: torch.Tensor,
        evidence_mask: torch.Tensor,
        query_mask: torch.Tensor,
        unobs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the forward pass of the MLP.
        """
        if self.embedding is not None:
            x = self.embedding(evidence_data, evidence_mask, query_mask, unobs_mask)
        else:
            raise ValueError(
                "No embedding provided; this is not recommended. If you want to use an identity embedding, use the `IdentityEmbedding` class."
            )
        x = self.hidden_layers(x)
        output = self.output_layer(x)
        return output
