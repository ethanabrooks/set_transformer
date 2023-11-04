from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.utils.data

from tabular.grid_world_with_values import GridWorldWithValues as GridWorld
from utils import Transition


@dataclass(frozen=True)
class Sequence(ABC):
    grid_world: GridWorld
    transitions: Transition[torch.Tensor]

    @classmethod
    @abstractmethod
    def collect_data(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    def make(
        cls,
        grid_world_args: dict,
        n_data: int,
        seed: int,
        **kwargs,
    ):
        # 2D deltas for up, down, left, right
        grid_world = GridWorld.make(**grid_world_args, n_tasks=n_data, seed=seed)

        transitions: Transition[torch.Tensor] = cls.collect_data(
            **kwargs, grid_world=grid_world
        )
        return cls(grid_world=grid_world, transitions=transitions)
