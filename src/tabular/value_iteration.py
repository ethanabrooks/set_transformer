import math
from typing import Optional

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from tabular.grid_world import GridWorld


class ValueIteration(GridWorld):
    def __init__(self, atol: float = 0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.atol = atol

    def improve_policy(self, V: torch.Tensor, idxs: Optional[torch.Tensor] = None):
        # self.check_V(V)
        R = self.get_rewards(idxs)
        T = self.get_transitions(idxs)
        Q = R + self.gamma * (T * V[:, None, None]).sum(-1)
        Pi = torch.zeros((len(R), self.n_states, len(self.deltas)), device=T.device)
        Pi.scatter_(-1, Q.argmax(dim=-1, keepdim=True), 1.0)
        return Pi

    def visualize_values(self, V: torch.Tensor, save_path: Optional[str] = None):
        dims = len(V.shape)
        global_min = V.min().item()
        global_max = V.max().item()

        def imshow(values: torch.Tensor, ax: plt.Axes):
            values = values[..., :-1]
            im = ax.imshow(
                values.reshape((2 * self.grid_size, self.grid_size)),
                cmap="hot",
                interpolation="nearest",
                vmin=global_min,
                vmax=global_max,
            )
            ax.axis("off")  # Turn off the axes
            return im

        if dims == 2:
            fig, ax = plt.subplots(figsize=(self.grid_size, self.grid_size))
            imshow(V, ax)

        elif dims == 3:
            n_tasks = V.shape[0]
            fig, axes = plt.subplots(
                1, n_tasks, figsize=(self.grid_size * n_tasks, self.grid_size)
            )
            if n_tasks == 1:
                axes = [axes]
            for idx, ax in enumerate(axes):
                imshow(V[idx], ax)

        elif dims == 4:
            n_rows, n_cols = V.shape[:2]
            # Adjust the subplot size depending on the number of rows
            fig_size = (self.grid_size / (n_rows**0.5)) * n_cols, (
                self.grid_size / (n_rows**0.5)
            ) * n_rows
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=fig_size, sharex="all", sharey="all"
            )

            # Capture the return image for colorbar
            ims = []
            for i in range(n_rows):
                for j in range(n_cols):
                    im = imshow(V[i, j], axes[i, j])
                    ims.append(im)

            # Add a single colorbar for all plots
            # fig.colorbar(ims[0], ax=axes.ravel().tolist())

        else:
            raise ValueError(f"Unsupported number of dimensions: {dims}")

        if save_path:
            plt.savefig(
                save_path, dpi=self.grid_size**2 / 2
            )  # adjust dpi if necessary
        return fig


def imshow(values: torch.Tensor, ax: plt.Axes, grid_size: int):
    global_min = values.min().item()
    global_max = values.max().item()

    values = values.reshape((grid_size, grid_size))
    im = ax.imshow(
        values, cmap="hot", interpolation="nearest", vmin=global_min, vmax=global_max
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# whitelist:
ValueIteration.visualize_values
