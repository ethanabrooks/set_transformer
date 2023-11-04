from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    def make(cls, sequence: Sequence):
        Q = cls.compute_values(sequence)
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
        metrics: dict,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ):
        return metrics, outputs, targets
