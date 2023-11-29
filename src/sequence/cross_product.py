from dataclasses import dataclass
from typing import Optional

import torch

from grid_world.base import GridWorld
from sequence.base import Sequence as BaseSequence
from utils import Transition


@dataclass(frozen=True)
class Sequence(BaseSequence):
    @classmethod
    def collect_data(
        cls, grid_world: GridWorld, omit_states_actions: int, partial_observation: bool
    ):
        a = grid_world.n_actions
        s = grid_world.n_states
        b = grid_world.n_tasks
        states = torch.arange(s).repeat_interleave(a)
        states = states[None].tile(b, 1)
        actions = torch.arange(a).repeat(s)
        actions = actions[None].tile(b, 1)
        action_probs = grid_world.Pi.repeat_interleave(a, 1)
        next_states, rewards, done, _ = grid_world.step_fn(
            states=states, actions=actions, time_remaining=None
        )

        b = grid_world.n_tasks
        s = grid_world.n_states
        a = len(grid_world.deltas)
        permutation = torch.rand(b, s * a).argsort(dim=1)

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
