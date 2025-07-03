import abc
from typing import Dict

import torch
from torch.utils.data import DataLoader


class BaseInferenceModule(abc.ABC):
    """Abstract base class for all inference modules."""

    @abc.abstractmethod
    def run(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
