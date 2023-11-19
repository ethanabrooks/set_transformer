from dataclasses import dataclass

import numpy as np

from dataset.base import Dataset as BaseDataset
from models.value_unconditional import DataPoint
from values.base import Values


@dataclass(frozen=True)
class Dataset(BaseDataset):
    max_n_bellman: int

    def __getitem__(self, idx) -> DataPoint:
        idx, n_bellman = np.unravel_index(idx, (len(self.sequence), self.max_n_bellman))
        transitions = self.sequence.transitions[idx]

        return DataPoint(
            action_probs=transitions.action_probs,
            actions=transitions.actions,
            idx=idx,
            n_bellman=n_bellman,
            next_states=transitions.next_states,
            q_values=self.Q[idx],
            rewards=transitions.rewards,
            states=transitions.states,
        )

    def __len__(self):
        return len(self.sequence) * self.max_n_bellman

    @classmethod
    def make(cls, values: Values, **kwargs):
        Q = values.Q
        return cls(**kwargs, max_n_bellman=len(Q), Q=Q.permute(1, 2, 0), values=values)
