from dataclasses import dataclass

import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.cm import viridis
from matplotlib.colors import ListedColormap, Normalize

from models.trajectories import MiniWorldModel
from sequence.grid_world_base import Sequence
from train.trajectories.base import Trainer as Base

assert isinstance(viridis, ListedColormap)


def plot_trajectory(
    box: torch.Tensor,
    done: torch.Tensor,
    pos: torch.Tensor,
    dir_vec: torch.Tensor,
    q_vals: torch.Tensor,
    rewards: torch.Tensor,
):
    [ep_boundaries] = done.nonzero(as_tuple=True)
    if len(ep_boundaries) == 0:
        return

    fig: plt.Figure
    axes: list[plt.Axes]
    fig, axes = plt.subplots(1, len(ep_boundaries), figsize=(6 * len(ep_boundaries), 6))
    if len(ep_boundaries) == 1:
        axes = [axes]

    ep_start = 0
    norm_q = Normalize(vmin=0, vmax=1)
    norm_rewards = Normalize(vmin=0, vmax=1)
    for ax, ep_boundary in zip(axes, ep_boundaries):
        episode_pos = pos[ep_start : ep_boundary + 1]
        x, y = episode_pos.T
        ax.plot(x, y)

        episode_dir = dir_vec[ep_start : ep_boundary + 1]
        dx, dy = 0.1 * episode_dir.T
        episode_q = q_vals[ep_start : ep_boundary + 1]
        episode_rewards = rewards[ep_start : ep_boundary + 1]

        # Normalize Q and rewards for color mapping

        for x, y, dx, dy, q, r in zip(x, y, dx, dy, episode_q, episode_rewards):
            color_q = viridis(norm_q(q))
            color_r = viridis(norm_rewards(r))
            # Arrow for Q-value (line)
            ax.arrow(
                x,
                y,
                dx,
                dy,
                head_width=0.2,
                head_length=0.2,
                fc=color_q,
                ec="black",
            )
            # Arrow for reward (head)
            ax.arrow(
                x,
                y,
                dx,
                dy,
                head_width=0.2,
                head_length=0.2,
                fc=color_r,
                ec="black",
                length_includes_head=True,
            )

        episode_goal = box[ep_boundary]
        ax.scatter(*episode_goal, color="red")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)

        ep_start = ep_boundary + 1
    return fig


def plot_trajectories(
    done: torch.Tensor, Q: torch.Tensor, rewards: torch.Tensor, states: torch.Tensor
):
    b, l = Q.shape
    assert [*done.shape] == [b, l]
    assert [*rewards.shape] == [b, l]
    assert [*states.shape] == [b, l, 4 * 3]
    states = states.reshape(b, l, 4, 3)
    states = states[:, :, [[x] for x in range(4)], [[0, 2]]]
    box, pos, dir_vec, _ = states.unbind(-2)

    for box, pos, dir_vec, done, q_vals, rewards in zip(
        box, pos, dir_vec, done, Q, rewards
    ):
        fig = plot_trajectory(
            box=box, done=done, pos=pos, dir_vec=dir_vec, q_vals=q_vals, rewards=rewards
        )
        if fig is None:
            continue
        yield fig


@dataclass(frozen=True)
class Trainer(Base):
    sequence: Sequence

    @classmethod
    def build_model(cls, **kwargs):
        return MiniWorldModel(**kwargs)

    def get_ground_truth(self, bellman_number: int):
        pass

    def update_plots(self, bellman_number: int, Q: torch.Tensor):
        transitions = self.sequence.transitions
        actions = transitions.actions[self.plot_indices]
        _, l = actions.shape
        for i, fig in enumerate(
            plot_trajectories(
                done=transitions.done[self.plot_indices],
                Q=Q[-1, self.plot_indices[:, None], torch.arange(l)[None], actions],
                rewards=transitions.rewards[self.plot_indices],
                states=transitions.next_states[self.plot_indices],
            )
        ):
            # fig.savefig(f"plot_{i}_bellman_{bellman_number}.png")
            if self.run is not None:
                self.run.log({f"plot {i}/bellman {bellman_number}": wandb.Image(fig)})
