from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from sequence.base import Sequence


@dataclass(frozen=True)
class Values(ABC):
    sequence: Sequence
    optimally_improved_policy_values: torch.Tensor
    Q: torch.Tensor
    stop_at_rmse: float

    @classmethod
    @abstractmethod
    def compute_values(cls, sequence: Sequence):
        raise NotImplementedError

    @classmethod
    def make(cls, sequence: Sequence, Q: Optional[torch.Tensor] = None, **kwargs):
        if Q is None:
            Q: torch.Tensor = cls.compute_values(sequence=sequence, **kwargs)
            q, b, _, _ = Q.shape
            Q = Q[
                torch.arange(q)[:, None, None],
                torch.arange(b)[None, :, None],
                sequence.transitions.states[None],
            ]
        stop_at_rmse = sequence.grid_world.stop_at_rmse
        return cls(
            sequence=sequence,
            optimally_improved_policy_values=sequence.grid_world.optimally_improved_policy_values,
            Q=Q,
            stop_at_rmse=stop_at_rmse,
        )

    def get_metrics(
        self,
        idxs: torch.Tensor,
        outputs: torch.Tensor,
    ):
        return dict()
