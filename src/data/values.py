from dataclasses import dataclass

import torch

from data.mdp import MDP


@dataclass(frozen=True)
class Values:
    mdp: MDP
    optimally_improved_policy_values: torch.Tensor
    Q: torch.Tensor
    stop_at_rmse: float

    @classmethod
    def make(cls, mdp: MDP, stop_at_rmse: float):
        Q = mdp.compute_values(stop_at_rmse)
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
