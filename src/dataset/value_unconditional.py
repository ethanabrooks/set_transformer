from dataclasses import dataclass

import torch
import numpy as np

from dataset.base import Dataset as BaseDataset
from utils import DataPoint
from values.bootstrap import Values


@dataclass(frozen=True)
class Dataset(BaseDataset):
    max_n_bellman: int
    Q: torch.Tensor
    V: torch.Tensor
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

        return DataPoint(
            action_probs=transitions.action_probs,
            actions=transitions.actions,
            idx=idx,
            n_bellman=n_bellman,
            next_obs=next_obs,
            next_states=transitions.next_states,
            obs=obs,
            q_values=self.Q[idx],
            rewards=transitions.rewards,
            states=transitions.states,
            values=self.V[idx, :, n_bellman],
        )

    def __len__(self):
        return len(self.sequence) * self.max_n_bellman

    @classmethod
    def make(cls, values: Values, **kwargs):
        return cls(
            **kwargs,
            max_n_bellman=len(values.Q),
            Q=values.Q.permute(1, 2, 0),
            V=values.V.permute(1, 2, 0),
            values=values,
        )
