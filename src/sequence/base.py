from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.utils.data

from utils import Transition


@dataclass(frozen=True)
class Sequence(ABC):
    gamma: float
    n_actions: int
    pad_value: int
    transitions: Transition[torch.Tensor]

    @classmethod
    @abstractmethod
    def make(cls, **kwargs):
        pass

    @property
    def n_tokens(self):
        return 1 + int(self.pad_value)

    def __len__(self):
        return len(self.transitions)
