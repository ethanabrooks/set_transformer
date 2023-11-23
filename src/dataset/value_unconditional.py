from dataclasses import dataclass

from dataset.base import Dataset as BaseDataset
from values.bootstrap import Values


@dataclass(frozen=True)
class Dataset(BaseDataset):
    values: Values

    def get_max_n_bellman(self):
        return len(self.values.Q)

    def input_q(self, idx: int, n_bellman: int):
        q_idx = max(0, 1 + n_bellman - self.bellman_delta)
        return self.values.bootstrap_Q[q_idx, idx]

    def target_q(self, idx: int, n_bellman: int):
        return self.values.Q[n_bellman, idx]
