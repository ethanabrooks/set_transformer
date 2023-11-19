from dataclasses import dataclass, replace

import torch

from sequence.base import Sequence as BaseSequence
from tabular.grid_world import GridWorld
from tabular.grid_world_with_values import GridWorldWithValues
from utils import Transition


@dataclass(frozen=True)
class Sequence(BaseSequence):
    @classmethod
    def collect_data(cls, grid_world: GridWorld, **kwargs):
        steps = grid_world.get_trajectories(**kwargs)
        states = grid_world.convert_2d_to_1d(steps.states).long()
        actions = steps.actions.squeeze(-1).long()
        next_states = grid_world.convert_2d_to_1d(steps.next_states).long()
        rewards = steps.rewards.squeeze(-1)
        action_probs = steps.action_probs

        states_actions = torch.stack([states, actions])
        terminal_states_actions = states_actions[:, :, -1]
        terminal_states_actions = states_actions == terminal_states_actions[..., None]
        terminal_states_actions = terminal_states_actions.all(0)
        steps.done[terminal_states_actions] = True

        assert steps.done[:, -1].all()

        return Transition(
            states=states,
            actions=actions,
            action_probs=action_probs,
            next_states=next_states,
            rewards=rewards,
            done=steps.done,
        )

    @classmethod
    def make(cls, grid_world: GridWorld, stop_at_rmse: float, **kwargs):
        transitions: Transition[torch.Tensor] = cls.collect_data(
            **kwargs, grid_world=grid_world
        )
        terminal_transitions = torch.zeros(
            (grid_world.n_tasks, grid_world.n_states, grid_world.n_actions),
            dtype=torch.bool,
        )
        terminal_transitions[
            torch.arange(grid_world.n_tasks)[:, None],
            transitions.states,
            transitions.actions,
        ] = transitions.done
        grid_world = replace(grid_world, terminal_transitions=terminal_transitions)
        grid_world = GridWorldWithValues.make(
            grid_world=grid_world, stop_at_rmse=stop_at_rmse
        )
        return cls(grid_world=grid_world, transitions=transitions)
