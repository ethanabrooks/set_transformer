from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib.axes import Axes

from models.trajectories import GridWorldModel, MiniWorldModel
from ppo.envs.envs import EnvType
from sequence.grid_world_base import Sequence
from train.plot import plot_grid_world_q_values, plot_trajectories
from train.trajectories.base import Trainer as Base
from values.bootstrap import Values


@dataclass(frozen=True)
class Trainer(Base):
    sequence: Sequence

    @classmethod
    def build_model(cls, env_type: EnvType, **kwargs):
        if env_type == EnvType.GRID_WORLD:
            return GridWorldModel(**kwargs)
        else:
            return MiniWorldModel(**kwargs)

    def get_ground_truth(self):
        pass

    def update_plots(self, bellman_number: int, Q: torch.Tensor):
        log_table = None
        if self.env_type == EnvType.SEQUENCE:
            transitions = self.sequence.transitions
            actions = transitions.actions[self.plot_indices]
            _, l = actions.shape
            log_table = {}
            for i, fig in enumerate(
                plot_trajectories(
                    done=transitions.done[self.plot_indices],
                    Q=Q[-1, self.plot_indices[:, None], torch.arange(l)[None], actions],
                    rewards=transitions.rewards[self.plot_indices],
                    states=transitions.next_states[self.plot_indices],
                )
            ):
                log_table[f"plot {i}/bellman {bellman_number}"] = wandb.Image(fig)
        elif self.env_type == EnvType.GRID_WORLD:
            q_targs: torch.Tensor = Values.compute_values(
                bootstrap_Q=Q[:, self.plot_indices],
                sequence=self.sequence[self.plot_indices],
            )
            q_targs = q_targs[-1]
            q_preds = Q[-1, self.plot_indices]
            states = self.sequence.transitions.states[self.plot_indices].long()
            actions = self.sequence.transitions.actions[self.plot_indices]
            fig, axs = plt.subplots(1, len(self.plot_indices), figsize=(15, 5))
            ax: Axes
            for i, q_targ, q_pred, ax, states, actions in zip(
                self.plot_indices, q_targs, q_preds, axs, states, actions
            ):
                n_states = self.sequence.observation_space.n
                q_pred_per_state = torch.empty(n_states, Q.size(-1))
                q_pred_per_state[states] = q_pred
                q_targ_per_state = torch.zeros_like(q_pred_per_state)
                q_targ_per_state[states, actions] = q_targ
                data = torch.cat([q_pred_per_state, q_targ_per_state], dim=0)
                grid_size: float = n_states**0.5
                assert grid_size.is_integer()
                plot_grid_world_q_values(ax=ax, q_values=data, grid_size=grid_size)
                ax.set_title(f"Agent {i+1}")
                log_table = {f"plot {i}/bellman {bellman_number}": wandb.Image(fig)}

        if log_table is not None and self.run is not None:
            self.run.log(log_table)
