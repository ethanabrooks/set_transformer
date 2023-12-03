from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.utils.data

from grid_world.base import GridWorld
from grid_world.values import GridWorldWithValues
from sequence.base import Sequence as Base
from utils import Transition


@dataclass(frozen=True)
class Sequence(Base, ABC):
    grid_world: GridWorldWithValues

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
            grid_world=grid_world, stop_at_rmse=stop_at_rmse, verbose=True
        )
        return cls(
            gamma=grid_world.gamma, grid_world=grid_world, transitions=transitions
        )

    @property
    def max_discrete_value(self):
        transitions = self.transitions
        pad_value: torch.Tensor = 1 + max(
            transitions.actions.max(),
            transitions.next_states.max(),
            transitions.rewards.max(),
            transitions.states.max(),
        )
        return pad_value.item()
