from dataclasses import dataclass

import torch

from models.trajectories import MiniWorldModel
from sequence.grid_world_base import Sequence
from train.trainer.base import Trainer as Base


@dataclass(frozen=True)
class Trainer(Base):
    sequence: Sequence

    @classmethod
    def build_model(cls, **kwargs):
        return MiniWorldModel(**kwargs)

    def get_ground_truth(self, bellman_number: int):
        pass

    def update_plots(self, bellman_number: int, Q: torch.Tensor):
        pass
