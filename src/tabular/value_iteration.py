import math
from typing import Optional

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from tabular.grid_world import GridWorld


def round_tensor(tensor: torch.Tensor, round_to: int):
    return (tensor * round_to).round().long()


class ValueIteration(GridWorld):
    def __init__(self, atol: float = 0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atol = atol

    def improve_policy(self, V: torch.Tensor, idxs: Optional[torch.Tensor] = None):
        # self.check_V(V)
        R = self.R
        T = self.T
        if idxs is not None:
            T = T.to(idxs.device)
            T = T[idxs]
            R = R.to(idxs.device)
            R = R[idxs]
        Q = R + self.gamma * (T * V[:, None, None]).sum(-1)
        Pi = torch.zeros(
            (self.n_tasks, self.n_states, len(self.deltas)), device=T.device
        )
        Pi.scatter_(-1, Q.argmax(dim=-1, keepdim=True), 1.0)
        return Pi

    def value_iteration(
        self,
        n_pi_bins: int,
        n_rounds: int,
        pi_lower_bound: float,
        stop_at_rmse: float,
    ):
        B = self.n_tasks
        S = self.n_states
        A = len(self.deltas)
        states = torch.tensor(
            [[i, j] for i in range(self.grid_size) for j in range(self.grid_size)]
        )
        alpha = torch.ones(len(self.deltas))
        Pi = (
            torch.distributions.Dirichlet(alpha)
            .sample((self.n_tasks, S))
            .tile(math.ceil(B / self.n_tasks), 1, 1)[:B]
        )
        Pi = round_tensor(Pi, n_pi_bins) / n_pi_bins
        Pi = Pi.float()
        Pi = torch.clamp(Pi, pi_lower_bound, 1)
        Pi = Pi / Pi.sum(-1, keepdim=True)
        self.check_pi(Pi)

        # Compute next states for each action and state for each batch (goal)
        next_states = states[:, None] + self.deltas[None, :]
        next_states = torch.clamp(next_states, 0, self.grid_size - 1)
        S_ = (
            next_states[..., 0] * self.grid_size + next_states[..., 1]
        )  # Convert to indices

        # Determine if next_state is the goal for each batch (goal)
        is_goal = (self.goals[:, None] == states[None]).all(-1)

        # Modify transition to go to absorbing state if the next state is a goal
        absorbing_state_idx = S - 1
        S_ = S_[None].tile(B, 1, 1)
        if self.absorbing_state:
            S_[is_goal[..., None].expand_as(S_)] = absorbing_state_idx

        # Insert row for absorbing state
        padding = (0, 0, 0, 1)  # left 0, right 0, top 0, bottom 1
        S_ = F.pad(S_, padding, value=absorbing_state_idx)
        R = is_goal.float()[..., None].tile(1, 1, A)
        R = F.pad(R, padding, value=0)  # Insert row for absorbing state

        V = None
        for _ in range(n_rounds):
            V_iter = self.evaluate_policy_iteratively(Pi, stop_at_rmse)
            V = self.evaluate_policy(Pi, V)
            # self.visualize_values(V)
            yield V_iter, Pi
            Pi = self.improve_policy(V)
            # self.visualize_policy(Pi)

    def visualize_values(self, V: torch.Tensor, task_idx: int = 0):
        global_min = V[task_idx].min().item()
        global_max = V[task_idx].max().item()

        if self.absorbing_state:
            V = V[:, :-1]
        values = V[task_idx].reshape((self.grid_size, self.grid_size))
        fig, ax = plt.subplots()
        im = ax.imshow(
            values,
            cmap="hot",
            interpolation="nearest",
            vmin=global_min,
            vmax=global_max,
        )

        # Add colorbar to each subplot
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig("values.png")


# whitelist:
ValueIteration.visualize_values
ValueIteration.value_iteration
