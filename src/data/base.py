import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

import torch
from torch.utils.data import Dataset


T = TypeVar("T")


class RLData(Dataset, ABC):
    def __len__(self):
        return len(self.discrete)

    def __getitem__(self, idx):
        return (
            idx,
            self.input_bellman[idx],
            self.continuous[idx],
            self.discrete[idx],
            *[v[idx] for v in self.values],
        )

    @property
    @abstractmethod
    def continuous(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def discrete(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def input_bellman(self) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def max_n_bellman(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def values(self) -> list[torch.Tensor]:
        raise NotImplementedError


def make(path: "str | Path", *args, **kwargs) -> RLData:
    path = Path(path)
    name = path.stem
    name = ".".join(path.parts)
    module = importlib.import_module(name)
    data: RLData = module.RLData(*args, **kwargs)
    return data
