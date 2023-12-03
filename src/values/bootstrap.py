from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sequence.base import Sequence
from values.base import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    bootstrap_Q: torch.Tensor

    @classmethod
    def compute_values(cls, bootstrap_Q: torch.Tensor, sequence: Sequence):
        grid_world = sequence.grid_world
        _, b, _, _ = bootstrap_Q.shape
        Pi = grid_world.Pi[torch.arange(b)[:, None], sequence.transitions.next_states]
        bootstrap_Q = F.pad(bootstrap_Q[:, :, 1:], (0, 0, 0, 1))
        Pi = Pi[None].expand_as(bootstrap_Q)
        R = sequence.transitions.rewards[None, ...]
        done = sequence.transitions.done[None, ...]
        Q = R + grid_world.gamma * ~done * (bootstrap_Q * Pi).sum(-1)
        return Q

    @classmethod
    def make(cls, bootstrap_Q: torch.Tensor, sequence: Sequence, **kwargs):
        Q: torch.Tensor = cls.compute_values(
            bootstrap_Q=bootstrap_Q, sequence=sequence, **kwargs
        )
        return cls(
            bootstrap_Q=bootstrap_Q,
            optimally_improved_policy_values=sequence.grid_world.optimally_improved_policy_values,
            sequence=sequence,
            Q=Q,
        )
