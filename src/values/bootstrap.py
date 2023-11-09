from dataclasses import dataclass

import torch

from sequence.base import Sequence
from tabular.grid_world import evaluate_policy
from values.base import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    @classmethod
    def compute_values(cls, bootstrap_Q: torch.Tensor, sequence: Sequence, **_):
        grid_world = sequence.grid_world
        q, b, s, a = bootstrap_Q.shape
        Pi = grid_world.Pi[None].expand_as(bootstrap_Q).reshape(-1, s, a)
        Q = bootstrap_Q.reshape(-1, s, a)
        R = grid_world.rewards[None].expand_as(bootstrap_Q).reshape(-1, s, a)
        T = (
            grid_world.transition_matrix[None]
            .expand(q, b, s, a, s)
            .reshape(-1, s, a, s)
        )
        new_Q = evaluate_policy(gamma=grid_world.gamma, Pi=Pi, Q=Q, R=R, T=T)
        return new_Q.view(q, b, s, a)

    @classmethod
    def make(cls, sequence: Sequence, stop_at_rmse: float, **kwargs):
        Q: torch.Tensor = cls.compute_values(sequence=sequence, **kwargs)
        q, b, _, _ = Q.shape
        Q = Q[
            torch.arange(q)[:, None, None],
            torch.arange(b)[None, :, None],
            sequence.transitions.states[None],
        ]
        return cls(
            optimally_improved_policy_values=sequence.grid_world.optimally_improved_policy_values,
            sequence=sequence,
            stop_at_rmse=stop_at_rmse,
            Q=Q,
        )
