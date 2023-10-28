import data.base
from data.utils import Transition


class RLData(data.base.RLData):
    def collect_data(self, **kwargs):
        steps = self.grid_world.get_trajectories(**kwargs)
        states = self.grid_world.convert_2d_to_1d(steps.states).long()
        actions = steps.actions.squeeze(-1).long()
        next_states = self.grid_world.convert_2d_to_1d(steps.next_states).long()
        rewards = steps.rewards.squeeze(-1)
        return Transition(
            states=states,
            actions=actions,
            action_probs=steps.action_probs,
            next_states=next_states,
            rewards=rewards,
            done=steps.done,
        )
