from dataclasses import dataclass
from typing import Optional

import torch

from sequence.base import Sequence as BaseSequence
from tabular.grid_world import GridWorld
from utils import Transition


@dataclass(frozen=True)
class Sequence(BaseSequence):
    @classmethod
    def collect_data(
        cls, grid_world: GridWorld, omit_states_actions: int, partial_observation: bool
    ):
        A = grid_world.n_actions
        S = grid_world.n_states
        B = grid_world.n_tasks
        states = torch.arange(S).repeat_interleave(A)
        states = states[None].tile(B, 1)
        actions = torch.arange(A).repeat(S)
        actions = actions[None].tile(B, 1)
        action_probs = grid_world.Pi.repeat_interleave(A, 1)
        next_states, rewards, done, _ = grid_world.step_fn(states, actions)

        B = grid_world.n_tasks
        S = grid_world.n_states
        A = len(grid_world.deltas)
        permutation = torch.rand(B, S * A).argsort(dim=1)

        def permute(
            x: torch.Tensor,
            i: int,
            idxs: Optional[torch.Tensor] = None,
        ):
            p = permutation.to(x.device)
            if idxs is not None:
                p = p[idxs]
            for _ in range(i - 1):
                p = p[None]
            while p.dim() < x.dim():
                p = p[..., None]

            return torch.gather(x, i, p.expand_as(x))

        if omit_states_actions > 0:
            action_probs, states, actions, next_states, rewards, done = [
                permute(x, 1)[:, omit_states_actions:]
                for x in [action_probs, states, actions, next_states, rewards, done]
            ]

        return Transition(
            action_probs=action_probs,
            states=states,
            actions=actions,
            next_states=next_states,
            rewards=rewards,
            done=done,
        )
