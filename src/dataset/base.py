from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.utils.data
from torch.utils.data import Dataset as BaseDataset

from models.set_transformer import SetTransformer
from models.value_unconditional import DataPoint
from sequence.base import Sequence
from values.base import Values


@dataclass(frozen=True)
class Dataset(BaseDataset):
    bellman_delta: int
    sequence: Sequence
    values: Values

    @abstractmethod
    def __getitem__(self, idx) -> DataPoint:
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self, n_batch: int, net: SetTransformer, plot_indices: torch.Tensor, **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def get_metrics(
        self,
        accuracy_threshold: float,
        bellman_delta: int,
        iterations: int,
        net: SetTransformer,
        x: DataPoint,
    ):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def make(
        cls,
        max_initial_bellman: Optional[int],
        sequence: Sequence,
        values: Values,
    ):
        raise NotImplementedError

    @property
    def n_actions(self):
        return len(self.sequence.grid_world.deltas)

    @property
    def n_tokens(self):
        transitions = self.sequence.transitions
        return 1 + max(
            transitions.actions.max(),
            transitions.next_states.max(),
            transitions.rewards.max(),
            transitions.states.max(),
        )

    def __len__(self):
        return len(self.sequence)
