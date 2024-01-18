from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.utils.data
from gymnasium.spaces import Discrete

from utils.dataclasses import Transition


@dataclass(frozen=True)
class Sequence(ABC):
    action_space: Discrete
    gamma: float
    observation_space: Discrete
    pad_value: int
    transitions: Transition[torch.Tensor]

    @classmethod
    @abstractmethod
    def make(cls, **kwargs):
        pass

    @property
    def n_tokens(self):
        return 1 + int(self.pad_value)

    def __getitem__(self, item):
        return type(self)(
            action_space=self.action_space,
            gamma=self.gamma,
            observation_space=self.observation_space,
            pad_value=self.pad_value,
            transitions=self.transitions[item],
        )

    def __len__(self):
        return len(self.transitions)
