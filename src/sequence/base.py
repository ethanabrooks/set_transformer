from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.utils.data

from tabular.grid_world import GridWorld
from tabular.grid_world_with_values import GridWorldWithValues
from utils import Transition


@dataclass(frozen=True)
class Sequence(ABC):
    grid_world: GridWorldWithValues
    transitions: Transition[torch.Tensor]

    @classmethod
    @abstractmethod
    def collect_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def make(cls, grid_world: GridWorld, **kwargs):
        transitions: Transition[torch.Tensor] = cls.collect_data(
            **kwargs, grid_world=grid_world
        )
        return cls(grid_world=grid_world, transitions=transitions)

    def __len__(self):
        return len(self.transitions)
