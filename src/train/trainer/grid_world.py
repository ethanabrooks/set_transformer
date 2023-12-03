from dataclasses import dataclass

import torch

import wandb
from models.trajectories import DiscreteObsModel
from sequence.grid_world_base import Sequence
from train.trainer.base import Trainer as Base


@dataclass(frozen=True)
class Trainer(Base):
    sequence: Sequence

    @classmethod
    def build_model(cls, **kwargs):
        return DiscreteObsModel(**kwargs)

    def get_ground_truth(self, bellman_number: int):
        Q = self.sequence.grid_world.Q
        return Q[1 : 1 + bellman_number]  # omit Q_0 from ground_truth

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
