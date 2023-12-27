from dataclasses import dataclass, replace

import torch
import wandb

from models.trajectories import GridWorldModel
from sequence.grid_world_base import Sequence
from train.trajectories.base import Trainer as Base


@dataclass(frozen=True)
class Trainer(Base):
    sequence: Sequence

    @classmethod
    def build_model(cls, **kwargs):
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
        stacked = torch.stack(
            [q_per_state, grid_world.Q[bellman_number, self.plot_indices]]
        )

        Pi = grid_world.Pi[None, self.plot_indices]
        v_per_state: torch.Tensor = stacked * Pi
        v_per_state = v_per_state.sum(-1)
        v_per_state = torch.unbind(v_per_state, dim=1)
        for i, plot_value in enumerate(v_per_state):
            fig = grid_world.visualize_values(plot_value)
            if self.run is not None:
                self.run.log({f"plot {i}/bellman {bellman_number}": wandb.Image(fig)})
