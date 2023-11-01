from dataclasses import dataclass

from dataset.mdp import MDP
from values.base import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    @classmethod
    def compute_values(cls, mdp: MDP, stop_at_rmse: float):
        raise NotImplementedError
