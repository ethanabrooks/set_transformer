from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sequence.base import Sequence
from values.base import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    V: torch.Tensor

    @classmethod
    def compute_values(
        cls,
        bootstrap_Q: torch.Tensor,
        sample_from_trajectories: bool,
        sequence: Sequence,
    ):
        grid_world = sequence.grid_world
        q, b, _, a = bootstrap_Q.shape
        Pi = grid_world.Pi[torch.arange(b)[:, None], sequence.transitions.next_states]
        if sample_from_trajectories:
            bootstrap_Q = F.pad(bootstrap_Q[:, :, 1:], (0, 0, 0, 1))
        else:
            bootstrap_indexed = torch.zeros(q, b, sequence.grid_world.n_states, a)
            bootstrap_indexed[
                torch.arange(q)[:, None, None],
                torch.arange(b)[None, :, None],
                sequence.transitions.states[None],
            ] = bootstrap_Q
            bootstrap_Q = bootstrap_indexed[
                torch.arange(q)[:, None, None],
                torch.arange(b)[None, :, None],
                sequence.transitions.next_states[None],
            ]
        Pi = Pi[None].expand_as(bootstrap_Q)
        R = sequence.transitions.rewards[None, ...]
        done = sequence.transitions.done[None, ...]
        Q = R + grid_world.gamma * ~done * (bootstrap_Q * Pi).sum(-1)
        return Q

    @classmethod
    def make(
        cls,
        bootstrap_Q: torch.Tensor,
        sequence: Sequence,
        stop_at_rmse: float,
        **kwargs
    ):
        V: torch.Tensor = (bootstrap_Q * sequence.transitions.action_probs).sum(-1)
        Q: torch.Tensor = cls.compute_values(
            bootstrap_Q=bootstrap_Q, sequence=sequence, **kwargs
        )
        return cls(
            optimally_improved_policy_values=sequence.grid_world.optimally_improved_policy_values,
            sequence=sequence,
            stop_at_rmse=stop_at_rmse,
            Q=Q,
            V=V,
        )
