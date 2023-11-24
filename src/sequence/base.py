from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.utils.data

from grid_world.base import GridWorld
from grid_world.values import GridWorldWithValues
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
    def make(cls, grid_world: GridWorld, stop_at_rmse: float, **kwargs):
        transitions: Transition[torch.Tensor] = cls.collect_data(
            **kwargs, grid_world=grid_world
        )
        grid_world = GridWorldWithValues.make(
            grid_world=grid_world, stop_at_rmse=stop_at_rmse
        )
        return cls(grid_world=grid_world, transitions=transitions)

    def __len__(self):
        return len(self.transitions)
