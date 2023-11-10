from dataclasses import dataclass

import torch

from sequence.base import Sequence
from values.base import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    @classmethod
    def compute_values(cls, bootstrap_Q: torch.Tensor, sequence: Sequence):
        grid_world = sequence.grid_world
        q, b, _, a = bootstrap_Q.shape
        Pi = grid_world.Pi[torch.arange(b)[:, None], sequence.transitions.next_states]
        bootstrap_Q = bootstrap_Q[:, :, ::a]
        bootstrap_Q = bootstrap_Q[
            torch.arange(q)[:, None, None],
            torch.arange(b)[None, :, None],
            sequence.transitions.next_states[None],
        ]
        Pi = Pi[None].expand_as(bootstrap_Q)
        R = sequence.transitions.rewards[None, ...]
        Q = R + grid_world.gamma * (bootstrap_Q * Pi).sum(-1)
        return Q

    @classmethod
    def make(cls, sequence: Sequence, stop_at_rmse: float, **kwargs):
        Q: torch.Tensor = cls.compute_values(sequence=sequence, **kwargs)
        return cls(
            optimally_improved_policy_values=sequence.grid_world.optimally_improved_policy_values,
            sequence=sequence,
            stop_at_rmse=stop_at_rmse,
            Q=Q,
        )
