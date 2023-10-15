from abc import ABC
from typing import TypeVar

from torch.utils.data import Dataset

T = TypeVar("T")


class RLData(Dataset, ABC):
    def __len__(self):
        return len(self.discrete)

    def __getitem__(self, idx):
        return (
            self.input_n_bellman[idx],
            self.action_probs[idx],
            self.discrete[idx],
            *[v[idx] for v in self.values],
        )

    @property
    def max_n_bellman(self):
        return len(self.V) - 1
