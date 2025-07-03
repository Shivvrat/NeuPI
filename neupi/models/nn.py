from typing import List, Optional

import torch
import torch.nn as nn

from neupi.core.model import BaseNNModel
from neupi.registry import register


@register("nn_model")
class MLP(BaseNNModel):
    """
    A flexible Multi-Layer Perceptron (MLP) network.

    This module creates a fully connected neural network with configurable
    hidden layers, activation functions, batch normalization, and dropout.

    Args:
        input_size (int): The number of features in the input tensor.
        hidden_sizes (List[int]): A list where each element is the number of
                                  neurons in a hidden layer.
        output_size (int): The number of neurons in the output layer.
        hidden_activation (str): The activation function for hidden layers.
                                 Supported: 'relu', 'leaky_relu'.
        use_batchnorm (bool): If True, adds a BatchNorm1d layer after each
                              hidden activation. Defaults to True.
        dropout_rate (float): The dropout probability. If 0.0, no dropout is
                              applied. Defaults to 0.0.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        hidden_activation: str = "relu",
        use_batchnorm: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        if hidden_activation == "relu":
            activation_fn = nn.ReLU
        elif hidden_activation == "leaky_relu":
            activation_fn = nn.LeakyReLU
        else:
            raise ValueError("Unsupported activation function.")

        layers = []
        current_size = input_size
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

    def initialize_weights(self, nonlinearity: str = "relu"):
        """
        Initializes the weights of the network using appropriate methods.
        """
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(
                    layer.weight,
                    mode="fan_in",
                    nonlinearity=nonlinearity,
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the MLP.
        """
        x = self.hidden_layers(x)
        output = self.output_layer(x)
        return output
