from dataclasses import dataclass

import values.base
from data.mdp import MDP


@dataclass(frozen=True)
class Values(values.base.Values):
    @classmethod
    def compute_values(cls, mdp: MDP, stop_at_rmse: float):
        return mdp.compute_values(stop_at_rmse)
