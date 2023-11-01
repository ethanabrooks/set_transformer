from dataclasses import dataclass

from sequence.base import Sequence
from values.base import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    @classmethod
    def compute_values(cls, sequence: Sequence, stop_at_rmse: float):
        raise NotImplementedError
