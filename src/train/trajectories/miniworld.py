from dataclasses import dataclass

import torch
import wandb

from models.trajectories import MiniWorldModel
from sequence.grid_world_base import Sequence
from train.plot import plot_trajectories
from train.trajectories.base import Trainer as Base


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
