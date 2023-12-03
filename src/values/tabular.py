from dataclasses import dataclass

from sequence.grid_world_base import Sequence
from values.base import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    @classmethod
    def compute_values(cls, sequence: Sequence):
        return sequence.grid_world.Q
