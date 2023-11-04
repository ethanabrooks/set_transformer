from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.utils.data

from tabular.grid_world import GridWorld
from utils import Transition


@dataclass(frozen=True)
class Sequence(ABC):
    grid_world: GridWorld
    Pi: torch.Tensor
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
        A = grid_world.n_actions
        S = grid_world.n_states
        B = n_data

        alpha = torch.ones(A)
        Pi: torch.Tensor = torch.distributions.Dirichlet(alpha).sample(
            (B, S)
        )  # random policies
        assert [*Pi.shape] == [B, S, A]

        print("Policy evaluation...")
        # states, actions, next_states, rewards = self.collect_data(**kwargs, Pi=Pi)
        transitions: Transition[torch.Tensor] = cls.collect_data(
            **kwargs, grid_world=grid_world, Pi=Pi
        )
        return cls(
            grid_world=grid_world,
            Pi=Pi,
            transitions=transitions,
        )
