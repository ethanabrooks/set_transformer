import torch

import data.base


class RLData(data.base.RLData):
    def collect_data(self, Pi):
        grid_world = self.grid_world
        A = len(grid_world.deltas)
        S = grid_world.n_states
        B = self.n_data
        states = torch.arange(S).repeat_interleave(A)
        states = states[None].tile(B, 1)
        actions = torch.arange(A).repeat(S)
        actions = actions[None].tile(B, 1)
        next_states, rewards, _, _ = self.grid_world.step_fn(states, actions)
        return states, actions, next_states, rewards
