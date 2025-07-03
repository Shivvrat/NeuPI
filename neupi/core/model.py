import abc

import torch.nn as nn


class BaseProbModel(nn.Module):
    """Abstract base class for all probabilistic models in NeuPI."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # Add other common methods or properties here


class BaseNNModel(nn.Module):
    """Abstract base class for all neural network models in NeuPI."""

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # Add other common methods or properties here
