from dataclasses import dataclass

from dataset.trajectories import Dataset as BaseDataset
from models.trajectories import DataPoint


@dataclass(frozen=True)
class Dataset(BaseDataset):
    def __getitem__(self, idx) -> DataPoint:
        transitions = self.sequence.transitions[idx]
        n_bellman = self.get_max_n_bellman() - 1

        obs = transitions.obs
        if obs is None:
            obs = transitions.states

        next_obs = transitions.next_obs
        if next_obs is None:
            next_obs = transitions.next_states

        return DataPoint(
            action_probs=transitions.action_probs,
            actions=transitions.actions,
            done=transitions.done,
            idx=idx,
            input_q=self.input_q(idx, n_bellman),
            n_bellman=n_bellman,
            next_obs=next_obs,
            obs=obs,
            rewards=transitions.rewards,
            target_q=self.target_q(idx, n_bellman),
        )

    def __len__(self):
        return len(self.sequence)
