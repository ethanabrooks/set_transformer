from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from gymnasium.spaces import Discrete

from grid_world.base import GridWorld
from grid_world.values import GridWorldWithValues
from sequence.grid_world_base import Sequence as BaseSequence
from sequence.grid_world_base import max_discrete_value
from utils.dataclasses import Transition


@dataclass(frozen=True)
class Sequence(BaseSequence):
    @classmethod
    def collect_data(cls, grid_world: GridWorld, partial_observation: bool, **kwargs):
        steps = grid_world.get_trajectories(**kwargs)
        states = steps.states.long()
        actions = steps.actions.squeeze(-1).long()
        next_states = steps.next_states.long()
        rewards = steps.rewards.squeeze(-1)
        action_probs = steps.action_probs

        states_actions = torch.stack([states, actions])
        terminal_states_actions = states_actions[:, :, -1]
        terminal_states_actions = states_actions == terminal_states_actions[..., None]
        terminal_states_actions = terminal_states_actions.all(0)
        steps.done[terminal_states_actions] = True
        first_step = F.pad(steps.done, (1, 0), value=True)[:, :-1]
        if partial_observation:
            assert not grid_world.is_wall.any()
            obs = 1 + steps.states
            obs = first_step[..., None] * obs
            next_obs = torch.zeros_like(obs)
        else:
            obs = next_obs = None

        assert steps.done[:, -1].all()

        return Transition(
            states=states,
            actions=actions,
            action_probs=action_probs,
            next_obs=next_obs,
            next_states=next_states,
            obs=obs,
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
            grid_world=grid_world, stop_at_rmse=stop_at_rmse, verbose=True
        )
        return cls(
            action_space=Discrete(grid_world.n_actions),
            gamma=grid_world.gamma,
            grid_world=grid_world,
            observation_space=Discrete(grid_world.n_states),
            pad_value=max_discrete_value(transitions) + 1,
            transitions=transitions,
        )
