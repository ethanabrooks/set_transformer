import torch

import data.base
from data.utils import Transition


class RLData(data.base.RLData):
    def collect_data(self, Pi: torch.Tensor):
        grid_world = self.grid_world
        A = len(grid_world.deltas)
        S = grid_world.n_states
        B = self.n_data
        states = torch.arange(S).repeat_interleave(A)
        states = states[None].tile(B, 1)
        actions = torch.arange(A).repeat(S)
        actions = actions[None].tile(B, 1)
        action_probs = Pi.repeat_interleave(A, 1)
        next_states, rewards, done, _ = self.grid_world.step_fn(states, actions)
        return Transition(
            states=states,
            actions=actions,
            action_probs=action_probs,
            next_states=next_states,
            rewards=rewards,
            done=done,
        )

    def get_n_metrics(self, *args, idxs: torch.Tensor, **kwargs):
        metrics, outputs = super().get_n_metrics(*args, idxs=idxs, **kwargs)
        if self.omit_states_actions == 0:
            values = outputs[
                -1,  # last iteration of policy evaluation
                0,  # output, not target
                :,
                :: len(self.grid_world.deltas),  # index into unique states
            ]

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
        return metrics, outputs
