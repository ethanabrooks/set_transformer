from dataclasses import dataclass

import torch

import data.base
from data.utils import Transition
from tabular.grid_world import GridWorld


@dataclass(frozen=True)
class MDP(data.base.MDP):
    @classmethod
    def collect_data(cls, grid_world: GridWorld, Pi: torch.Tensor, **kwargs):
        steps = grid_world.get_trajectories(**kwargs, Pi=Pi)
        states = grid_world.convert_2d_to_1d(steps.states).long()
        actions = steps.actions.squeeze(-1).long()
        next_states = grid_world.convert_2d_to_1d(steps.next_states).long()
        rewards = steps.rewards.squeeze(-1)
        action_probs = Pi[torch.arange(len(Pi))[:, None], states]
        return Transition(
            states=states,
            actions=actions,
            action_probs=action_probs,
            next_states=next_states,
            rewards=rewards,
            done=steps.done,
        )


class Dataset(data.base.Dataset):
    def make_mdp(self, *args, **kwargs):
        return MDP.make(*args, **kwargs)
