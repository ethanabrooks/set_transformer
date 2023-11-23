from dataclasses import dataclass

import numpy as np

from dataset.base import Dataset as BaseDataset
from models.value_unconditional import DataPoint
from values.bootstrap import Values


@dataclass(frozen=True)
class Dataset(BaseDataset):
    values: Values

    def __getitem__(self, idx) -> DataPoint:
        idx, n_bellman = np.unravel_index(idx, (len(self.sequence), self.max_n_bellman))
        transitions = self.sequence.transitions[idx]

        obs = transitions.obs
        if obs is None:
            obs = transitions.states

        next_obs = transitions.next_obs
        if next_obs is None:
            next_obs = transitions.next_states

        values = (
            self.values.bootstrap_Q[n_bellman, idx] * transitions.action_probs
        ).sum(-1)
        return DataPoint(
            action_probs=transitions.action_probs,
            actions=transitions.actions,
            idx=idx,
            n_bellman=n_bellman,
            next_obs=next_obs,
            next_states=transitions.next_states,
            obs=obs,
            q_values=self.values.Q[:, idx].T,
            rewards=transitions.rewards,
            states=transitions.states,
            values=values,
        )

    def __len__(self):
        return len(self.sequence) * self.max_n_bellman

    @property
    def max_n_bellman(self):
        return len(self.values.Q)
