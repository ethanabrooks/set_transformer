from dataclasses import dataclass


from dataset.base import Dataset as BaseDataset
from models.value_unconditional import DataPoint
from sequence.base import Sequence
from values.base import Values


@dataclass(frozen=True)
class Dataset(BaseDataset):
    max_n_bellman: int

    def __getitem__(self, idx) -> DataPoint:
        n_bellman = idx % self.max_n_bellman
        idx = idx // self.max_n_bellman
        transitions = self.sequence.transitions[idx]
        return DataPoint(
            action_probs=transitions.action_probs,
            actions=transitions.actions,
            idx=idx,
            n_bellman=n_bellman,
            next_states=transitions.next_states,
            q_values=self.Q[:, idx],
            rewards=transitions.rewards,
            states=transitions.states,
        )

    def __len__(self):
        return len(self.sequence) * self.max_n_bellman

    @classmethod
    def make(cls, sequence: Sequence, values: Values):
        Q = values.Q
        return cls(max_n_bellman=len(Q), Q=Q, sequence=sequence, values=values)
