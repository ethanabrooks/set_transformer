from dataclasses import dataclass

from sequence.base import Sequence as BaseSequence
from tabular.grid_world import GridWorld
from utils import Transition
from values.tabular import Values as BaseValues


@dataclass(frozen=True)
class Sequence(BaseSequence):
    @classmethod
    def collect_data(cls, grid_world: GridWorld, **kwargs):
        steps = grid_world.get_trajectories(**kwargs)
        states = grid_world.convert_2d_to_1d(steps.states).long()
        actions = steps.actions.squeeze(-1).long()
        next_states = grid_world.convert_2d_to_1d(steps.next_states).long()
        rewards = steps.rewards.squeeze(-1)
        return Transition(
            states=states,
            actions=actions,
            action_probs=steps.action_probs,
            next_states=next_states,
            rewards=rewards,
            done=steps.done,
        )


@dataclass(frozen=True)
class Values(BaseValues):
    pass
