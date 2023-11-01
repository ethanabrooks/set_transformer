from dataclasses import dataclass

import torch

from data.mdp import MDP as BaseMDP
from data.utils import Transition
from tabular.grid_world import GridWorld
from values.tabular import Values as BaseValues


@dataclass(frozen=True)
class MDP(BaseMDP):
    @classmethod
    def collect_data(cls, grid_world: GridWorld, Pi: torch.Tensor, **kwargs):
        steps = grid_world.get_trajectories(**kwargs, Pi=Pi)
        states = grid_world.convert_2d_to_1d(steps.states).long()
        actions = steps.actions.squeeze(-1).long()
        next_states = grid_world.convert_2d_to_1d(steps.next_states).long()
        rewards = steps.rewards.squeeze(-1)
        action_probs = steps.action_probs
        return Transition(
            states=states,
            actions=actions,
            action_probs=action_probs,
            next_states=next_states,
            rewards=rewards,
            done=steps.done,
        )


@dataclass(frozen=True)
class Values(BaseValues):
    pass
