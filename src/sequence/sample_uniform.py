from dataclasses import dataclass

import torch

from sequence.base import Sequence as BaseSequence
from tabular.grid_world import GridWorld
from utils import Transition


@dataclass(frozen=True)
class Sequence(BaseSequence):
    @classmethod
    def collect_data(cls, grid_world: GridWorld):
        A = grid_world.n_actions
        S = grid_world.n_states
        B = grid_world.n_tasks
        states = torch.arange(S).repeat_interleave(A)
        states = states[None].tile(B, 1)
        actions = torch.arange(A).repeat(S)
        actions = actions[None].tile(B, 1)
        action_probs = grid_world.Pi.repeat_interleave(A, 1)
        next_states, rewards, done, _ = grid_world.step_fn(states, actions)
        return Transition(
            states=states,
            actions=actions,
            action_probs=action_probs,
            next_states=next_states,
            rewards=rewards,
            done=done,
        )
