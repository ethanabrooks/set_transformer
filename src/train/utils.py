from typing import Optional

import torch
from matplotlib import pyplot as plt
from matplotlib.cm import hot, hsv
from matplotlib.colors import LinearSegmentedColormap, Normalize

assert isinstance(hot, LinearSegmentedColormap)
MAX_PLOTS = 30


def plot_trajectory(
    boxes: list[torch.Tensor],
    done: torch.Tensor,
    pos: torch.Tensor,
    dir_vec: torch.Tensor,
    q_vals: Optional[torch.Tensor],
    rewards: torch.Tensor,
):
    [ep_boundaries] = done.nonzero(as_tuple=True)
    if len(ep_boundaries) == 0:
        return
    ep_boundaries = ep_boundaries[:MAX_PLOTS]

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
        if q_vals is not None:
            episode_q = q_vals[ep_start : ep_boundary + 1]
            for x, y, dx, dy, q in zip(x, y, dx, dy, episode_q):
                color_q = hot(norm_q(q))
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
        episode_rewards = rewards[ep_start : ep_boundary + 1]

        # Normalize Q and rewards for color mapping

        for x, y, dx, dy, r in zip(x, y, dx, dy, episode_rewards):
            color_r = hot(norm_rewards(r))
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

        for i, box in enumerate(boxes):
            episode_goal = box[ep_boundary]
            color = hsv(i / len(boxes))
            ax.scatter(*episode_goal, color=color)
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
            boxes=[box],
            done=done,
            pos=pos,
            dir_vec=dir_vec,
            q_vals=q_vals,
            rewards=rewards,
        )
        if fig is None:
            continue
        yield fig
