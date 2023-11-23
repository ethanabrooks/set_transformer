from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
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
    def evaluate(
        self, n_batch: int, net: SetTransformer, plot_indices: torch.Tensor, **kwargs
    ):
        raise NotImplementedError

    @abstractmethod
    def get_max_n_bellman(self):
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

    @abstractmethod
    def input_q(self, idx: int, n_bellman: int):
        raise NotImplementedError

    @abstractmethod
    def target_q(self, idx: int, n_bellman: int):
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

    def __getitem__(self, idx) -> DataPoint:
        idx, n_bellman = np.unravel_index(
            idx, (len(self.sequence), self.get_max_n_bellman())
        )
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
            input_q=self.input_q(idx, n_bellman),
            n_bellman=n_bellman,
            next_obs=next_obs,
            next_states=transitions.next_states,
            obs=obs,
            rewards=transitions.rewards,
            states=transitions.states,
            target_q=self.target_q(idx, n_bellman),
        )

    def __len__(self):
        return len(self.sequence) * self.get_max_n_bellman()
