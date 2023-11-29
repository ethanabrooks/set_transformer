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
    first: bool = True
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

    def convert_2d_to_1d(self, state: torch.Tensor):
        return self.grid_world.convert_2d_to_1d(state)

    def reset(self):
        assert self.grid_world.n_tasks == 1
        current_state = self.grid_world.reset_fn()
        self.current_state = current_state.item()
        self.first = True
        self.time_remaining = self.time_limit
        return self.current_state

    def step(self, action: Union[torch.Tensor, int]):
        info = dict()
        if self.first:
            self.first = False
            if isinstance(self.grid_world, GridWorldWithValues):
                optimal = (
                    self.grid_world.optimally_improved_policy_values[
                        :, self.current_state
                    ]
                    .max()
                    .item()
                )
                info.update(optimal=optimal)
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
        i: dict
        r: torch.Tensor
        d: torch.Tensor
        current_state, r, d, i = self.grid_world.step_fn(
            states=current_state, actions=action, time_remaining=time_remaining
        )
        self.current_state = current_state.item()
        reward = r.item()
        done = d.item()
        info.update(i)
        return self.current_state, reward, done, info
