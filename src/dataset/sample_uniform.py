from dataclasses import dataclass

import torch

from dataset.mdp import MDP as BaseMDP
from dataset.utils import Transition
from tabular.grid_world import GridWorld
from values.tabular import Values as BaseValues


@dataclass(frozen=True)
class MDP(BaseMDP):
    @classmethod
    def collect_data(cls, grid_world: GridWorld, Pi: torch.Tensor):
        A = len(grid_world.deltas)
        S = grid_world.n_states
        B = grid_world.n_tasks
        states = torch.arange(S).repeat_interleave(A)
        states = states[None].tile(B, 1)
        actions = torch.arange(A).repeat(S)
        actions = actions[None].tile(B, 1)
        action_probs = Pi.repeat_interleave(A, 1)
        next_states, rewards, done, _ = grid_world.step_fn(states, actions)
        return Transition(
            states=states,
            actions=actions,
            action_probs=action_probs,
            next_states=next_states,
            rewards=rewards,
            done=done,
        )


@dataclass(frozen=True)
class Values(BaseValues):
    def get_metrics(self, idxs: torch.Tensor, outputs: torch.Tensor):
        grid_world = self.mdp.grid_world
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
