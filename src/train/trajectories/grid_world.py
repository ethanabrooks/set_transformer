from dataclasses import dataclass, replace

import torch
import wandb
from matplotlib import pyplot as plt

from models.trajectories import GridWorldModel
from sequence.grid_world_base import Sequence
from train.plot import plot_grid_world_q_values
from train.trajectories.base import Trainer as Base


@dataclass(frozen=True)
class Trainer(Base):
    sequence: Sequence

    @classmethod
    def build_model(cls, env_type, **kwargs):
        return GridWorldModel(**kwargs)

    @classmethod
    def make(cls, baseline: bool, sequence: Sequence, **kwargs):
        grid_world = sequence.grid_world
        if baseline:
            grid_world = replace(grid_world, Q=grid_world.Q[[0, -1]])
            sequence = replace(sequence, grid_world=grid_world)
        return super().make(baseline=baseline, **kwargs, sequence=sequence)

    def get_ground_truth(self):
        return self.sequence.grid_world.Q

    def update_plots(self, bellman_number: int, Q: torch.Tensor):
        grid_world = self.sequence.grid_world
        q_per_state = torch.empty(
            len(self.plot_indices), grid_world.n_states, len(grid_world.deltas)
        )
        q_per_state[
            torch.arange(len(self.plot_indices))[:, None],
            self.sequence.transitions.states[self.plot_indices],
        ] = Q[-1, self.plot_indices]
        try:
            stacked = torch.stack(
                [q_per_state, grid_world.Q[bellman_number, self.plot_indices]], dim=1
            )
        except IndexError:
            return
        Qs = stacked[:, None]
        grid_size = grid_world.grid_size

        for i, Q in enumerate(Qs):
            n_iterations, _, n_states, n_actions = Q.shape
            assert n_actions == grid_world.n_actions
            assert (
                n_states == grid_world.n_states - 1
                if grid_world.use_absorbing_state
                else grid_world.n_states
            )
            Q = Q.reshape(n_iterations, 2 * n_states, n_actions)
            Q = Q.cpu().numpy()

            fig, axes = plt.subplots(
                1, n_iterations, figsize=(grid_size * n_iterations, grid_size)
            )
            if n_iterations == 1:
                axes = [axes]
            for ax, q_values in zip(axes, Q):
                plot_grid_world_q_values(ax=ax, grid_size=grid_size, q_values=q_values)
            if self.run is not None:
                self.run.log({f"plot {i}/bellman {bellman_number}": wandb.Image(fig)})
