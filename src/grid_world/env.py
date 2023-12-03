from dataclasses import dataclass
from typing import Optional, Union

import torch
from gym.spaces import Discrete, MultiDiscrete

from envs.base import Env
from grid_world.values import GridWorld, GridWorldWithValues


@dataclass
class Env(Env):
    grid_world: GridWorld
    current_state: Optional[int] = None
    time_limit: Optional[int] = None
    time_remaining: Optional[int] = None

    @property
    def action_space(self):
        return Discrete(len(self.grid_world.deltas))

    @property
    def grid_size(self):
        return self.grid_world.grid_size

    @property
    def observation_space(self):
        return MultiDiscrete([self.grid_size, self.grid_size])

    @property
    def policy(self):
        return self.grid_world.Pi.squeeze(0)

    @property
    def values(self):
        if isinstance(self.grid_world, GridWorldWithValues):
            return self.grid_world.Q.squeeze(1)

    def convert_2d_to_1d(self, state: torch.Tensor):
        return self.grid_world.convert_2d_to_1d(state)

    def optimal(self, state: "int | torch.Tensor") -> Optional[float]:
        if isinstance(state, torch.Tensor):
            state = state.long().item()
        if isinstance(self.grid_world, GridWorldWithValues):
            return (
                self.grid_world.optimally_improved_policy_values[:, state].max().item()
            )

    def reset(self, state: Optional[torch.Tensor] = None):
        assert self.grid_world.n_tasks == 1
        if state is None:
            current_state = self.grid_world.reset_fn()
        else:
            current_state = state
        self.current_state = current_state.item()
        self.time_remaining = self.time_limit
        return self.current_state, {}

    def step(self, action: Union[torch.Tensor, int]):
        info = dict()
        if isinstance(action, int):
            action = torch.tensor([action])
        action = action.reshape(1)
        assert isinstance(self.current_state, int)
        current_state: torch.Tensor = torch.tensor([self.current_state])
        if self.time_remaining is None:
            time_remaining = None
        else:
            time_remaining = self.time_remaining
            self.time_remaining -= 1
        info: dict
        reward: torch.Tensor
        done: torch.Tensor
        current_state, reward, done, truncated, info = self.grid_world.step_fn(
            states=current_state, actions=action, time_remaining=time_remaining
        )
        self.current_state = current_state.item()
        reward = reward.item()
        done = done.item()
        info.update(info)
        return self.current_state, reward, done, truncated, info
