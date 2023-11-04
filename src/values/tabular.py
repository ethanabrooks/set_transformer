from dataclasses import dataclass

import torch

from sequence.base import Sequence
from values.base import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    @classmethod
    def compute_values(cls, sequence: Sequence, stop_at_rmse: float):
        grid_world = sequence.grid_world
        return torch.stack(
            list(
                grid_world.evaluate_policy_iteratively(
                    Pi=grid_world.Pi, stop_at_rmse=stop_at_rmse
                )
            )
        )
