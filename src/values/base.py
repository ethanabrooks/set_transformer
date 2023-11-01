from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from dataset.mdp import MDP


@dataclass(frozen=True)
class Values(ABC):
    mdp: MDP
    optimally_improved_policy_values: torch.Tensor
    Q: torch.Tensor
    stop_at_rmse: float

    @classmethod
    @abstractmethod
    def compute_values(cls, mdp: MDP, stop_at_rmse: float):
        raise NotImplementedError

    @classmethod
    def make(cls, mdp: MDP, stop_at_rmse: float):
        Q = cls.compute_values(mdp, stop_at_rmse)
        optimally_improved_policy_values = mdp.grid_world.evaluate_improved_policy(
            Q=Q[-1], stop_at_rmse=stop_at_rmse
        ).cuda()
        return cls(
            mdp=mdp,
            optimally_improved_policy_values=optimally_improved_policy_values,
            Q=Q,
            stop_at_rmse=stop_at_rmse,
        )

    def get_metrics(
        self,
        idxs: torch.Tensor,
        outputs: torch.Tensor,
    ):
        return dict()
