from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sequence.grid_world_base import Sequence
from sequence.grid_world_base import Sequence as GridWorldSequence
from values.base import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    bootstrap_Q: torch.Tensor

    @classmethod
    def compute_target_q(
        cls,
        bootstrap_Q: torch.Tensor,
        done: torch.Tensor,
        gamma: float,
        Pi: torch.Tensor,
        R: torch.Tensor,
    ):
        return R + gamma * ~done * (bootstrap_Q * Pi).sum(-1)

    @classmethod
    def compute_values(cls, bootstrap_Q: torch.Tensor, sequence: Sequence):
        Pi = F.pad(sequence.transitions.action_probs[:, 1:], (0, 0, 0, 1))
        bootstrap_Q = F.pad(bootstrap_Q[:, :, 1:], (0, 0, 0, 1))
        Pi = Pi[None].expand_as(bootstrap_Q)
        R = sequence.transitions.rewards[None, ...]
        done = sequence.transitions.done[None, ...]
        return cls.compute_target_q(
            bootstrap_Q=bootstrap_Q,
            done=done,
            gamma=sequence.gamma,
            Pi=Pi,
            R=R,
        )

    @classmethod
    def make(cls, bootstrap_Q: torch.Tensor, sequence: Sequence, **kwargs):
        Q: torch.Tensor = cls.compute_values(
            bootstrap_Q=bootstrap_Q, sequence=sequence, **kwargs
        )
        if isinstance(sequence, GridWorldSequence):
            optimally_improved_policy_values = (
                sequence.grid_world.optimally_improved_policy_values
            )
        else:
            optimally_improved_policy_values = None
        return cls(
            bootstrap_Q=bootstrap_Q,
            optimally_improved_policy_values=optimally_improved_policy_values,
            sequence=sequence,
            Q=Q,
        )


@dataclass(frozen=True)
class BellmanStarValues(Values):
    @classmethod
    def compute_target_q(
        cls,
        bootstrap_Q: torch.Tensor,
        done: torch.Tensor,
        gamma: float,
        Pi: torch.Tensor,
        R: torch.Tensor,
    ):
        return R + gamma * ~done * bootstrap_Q.max(-1).values
