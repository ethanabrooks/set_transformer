import torch

import data.base


class RLData(data.base.RLData):
    def collect_data(self, Pi):
        grid_world = self.grid_world
        A = len(grid_world.deltas)
        S = grid_world.n_states
        B = self.n_data
        states = torch.arange(S).repeat_interleave(A)
        states = states[None].tile(B, 1)
        actions = torch.arange(A).repeat(S)
        actions = actions[None].tile(B, 1)
        next_states, rewards, _, _ = self.grid_world.step_fn(states, actions)
        return states, actions, next_states, rewards

    def get_n_metrics(self, *args, idxs: torch.Tensor, **kwargs):
        metrics, plot_values, outputs = super().get_n_metrics(
            *args, idxs=idxs, **kwargs
        )
        if self.omit_states_actions == 0:
            values = outputs[:, :: len(self.grid_world.deltas)]
            improved_policy_value = self.compute_improved_policy_value(
                idxs=idxs, values=values
            )
            optimally_improved_policy_values = self.optimally_improved_policy_values[
                idxs
            ]
            regret = optimally_improved_policy_values - improved_policy_value
            metrics.update(
                improved_policy_value=improved_policy_value.mean().item(),
                regret=regret.mean().item(),
            )
        return metrics, plot_values, outputs
