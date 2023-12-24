from typing import Optional

import torch
from matplotlib import pyplot as plt
from matplotlib.cm import hot, hsv
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle

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

    def set_alpha(value: float):
        normalized = norm_q(value)
        clipped = max(0, min(1, normalized))
        return 0.2 + 0.8 * clipped

    for ax, ep_boundary in zip(axes, ep_boundaries):
        ax.set_xlim([0, 6])
        ax.set_ylim([0, 6])

        for i, box in enumerate(boxes):
            episode_goal = box[ep_boundary]
            color = hsv(i / len(boxes))
            radius = 1.7356854249492382
            circle = Circle(edgecolor=color, fill=False, radius=radius, xy=episode_goal)
            ax.add_patch(circle)  # Add circle to the plot

        episode_pos = pos[ep_start : ep_boundary + 1]
        xs, ys = episode_pos.T
        ax.plot(xs, ys)

        episode_dir = dir_vec[ep_start : ep_boundary + 1]
        dxs, dys = 0.1 * episode_dir.T
        if q_vals is not None:
            episode_q = q_vals[ep_start : ep_boundary + 1]
            for x, y, dx, dy, q in zip(xs, ys, dxs, dys, episode_q):
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
                    alpha=set_alpha(norm_q(q)),
                )
        episode_rewards = rewards[ep_start : ep_boundary + 1]

        # Normalize Q and rewards for color mapping

        for x, y, dx, dy, r in zip(xs, ys, dxs, dys, episode_rewards):
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
                alpha=set_alpha(norm_rewards(r)),
            )

        ep_start = ep_boundary + 1
    return fig


def plot_trajectories(
    done: torch.Tensor, Q: torch.Tensor, rewards: torch.Tensor, states: torch.Tensor
):
    b, l = Q.shape
    assert [*done.shape] == [b, l]
    assert [*rewards.shape] == [b, l]
    assert [*states.shape][:-1] == [b, l]
    states = states.reshape(b, l, -1, 3)[..., [0, 2]]
    *boxes, pos, dir_vec, _ = states.unbind(2)
    done[:, -1] = True

    for *boxes, pos, dir_vec, done, q_vals, rewards in zip(
        *boxes, pos, dir_vec, done, Q, rewards
    ):
        fig = plot_trajectory(
            boxes=boxes,
            done=done,
            pos=pos,
            dir_vec=dir_vec,
            q_vals=q_vals,
            rewards=rewards,
        )
        if fig is None:
            continue
        yield fig
