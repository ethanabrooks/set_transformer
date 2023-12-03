from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.utils.data

from utils import Transition


@dataclass(frozen=True)
class Sequence(ABC):
    gamma: float
    transitions: Transition[torch.Tensor]

    @classmethod
    @abstractmethod
    def make(cls, **kwargs):
        pass

    @property
    @abstractmethod
    def max_discrete_value(self):
        pass

    def __len__(self):
        return len(self.transitions)
