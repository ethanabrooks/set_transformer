from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import hot, hsv
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, Polygon

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


def plot_grid_world_q_values(ax: Axes, grid_size: int, q_values: np.ndarray):
    """
    Plots the state values as triangles in a grid with the hypotenuse placed diagonally.

    Args:
    data (numpy.array): A 2D array of shape (n x grid_size^2, 4) representing the values.
    ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
    """
    assert q_values.ndim == 2

    # Define the triangle coordinates relative to the center of each cell
    # The hypotenuse will now be diagonal rather than horizontal or vertical
    triangles = [
        np.array([[0.5, 0.5], [1, 0], [1, 1]]),  # Down-Right
        np.array([[0.5, 0.5], [0, 1], [0, 0]]),  # Up-Left
        np.array([[0.5, 0.5], [0, 0], [1, 0]]),  # Down-Left
        np.array([[0.5, 0.5], [1, 1], [0, 1]]),  # Up-Right
    ]

    for index, value in enumerate(q_values):
        row, col = divmod(index, grid_size)
        for v, triangle in zip(value, triangles):
            # Shift the triangle to the correct grid cell
            shifted_triangle = triangle + np.array([col, row])
            color = plt.cm.hot(v)
            patch = Polygon(shifted_triangle, closed=True, color=color)
            ax.add_patch(patch)

    y_size = len(q_values) / grid_size
    assert y_size.is_integer()
    ax.set_xlim([0, grid_size])
    ax.set_ylim([0, y_size])
    ax.set_aspect("equal", "box")
    ax.axis("off")


def plot_grid_world_values(  # noqa: Vulture
    ax: Axes, grid_size: int, values: np.ndarray, use_absorbing_state: bool
):

    n, _ = values.shape
    if use_absorbing_state:
        values = values[..., :-1]
    ax.imshow(
        values.reshape((n * grid_size, grid_size)),
        cmap="hot",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.axis("off")  # Turn off the axes
