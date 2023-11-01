from dataclasses import dataclass

import torch

from values.tabular import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    def get_metrics(self, idxs: torch.Tensor, outputs: torch.Tensor):
        grid_world = self.sequence.grid_world
        S = grid_world.n_states
        A = len(grid_world.deltas)
        _, _, L, _ = outputs.shape

        if L == S * A:
            values = outputs[
                -1,  # last iteration of policy evaluation
                :,
                ::A,  # index into unique states
            ]

            improved_policy_value = grid_world.evaluate_improved_policy(
                idxs=idxs, Q=values, stop_at_rmse=self.stop_at_rmse
            )
            optimally_improved_policy_values = self.optimally_improved_policy_values[
                idxs
            ]
            regret: torch.Tensor = (
                optimally_improved_policy_values - improved_policy_value
            )
            return dict(
                improved_policy_value=improved_policy_value.mean().item(),
                regret=regret.mean().item(),
            )
        else:
            return dict()
