from dataclasses import dataclass

import torch

from values.tabular import Values as BaseValues


@dataclass(frozen=True)
class Values(BaseValues):
    stop_at_rmse: float

    @classmethod
    def make(cls, stop_at_rmse: float, **kwargs):
        values = BaseValues.make(**kwargs)
        return cls(
            optimally_improved_policy_values=values.optimally_improved_policy_values,
            Q=values.Q,
            sequence=values.sequence,
            stop_at_rmse=stop_at_rmse,
        )

    def get_metrics(self, idxs: torch.Tensor, outputs: torch.Tensor):
        grid_world = self.sequence.grid_world
        s = grid_world.n_states
        a = len(grid_world.deltas)
        _, _, l, _ = outputs.shape

        if l == s * a:
            values = outputs[
                -1,  # last iteration of policy evaluation
                :,
                ::a,  # index into unique states
            ]

            improved_policy_value = grid_world[idxs].evaluate_improved_policy(
                Q=values, stop_at_rmse=self.stop_at_rmse
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
