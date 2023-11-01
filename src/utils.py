from dataclasses import dataclass
import torch
from typing import Generic, TypeVar
import numpy as np

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class Transition(Generic[T]):
    states: T
    actions: T
    action_probs: T
    next_states: T
    rewards: T
    done: T

    def __getitem__(self, idx: torch.Tensor):
        def index(x: torch.Tensor):
            if isinstance(idx, torch.Tensor):
                x = x.to(idx.device)
            return x[idx]

        return type(self)(
            states=index(self.states),
            actions=index(self.actions),
            action_probs=index(self.action_probs),
            next_states=index(self.next_states),
            rewards=index(self.rewards),
            done=index(self.done),
        )


def tensor_hash(tensor: torch.Tensor):
    # Ensure that the tensor is on the CPU and converted to 1D
    tensor_1d: torch.Tensor = tensor.cpu().flatten()
    array_1d: np.ndarray = tensor_1d.numpy()

    # Convert the 1D tensor to bytes and hash
    return hash(array_1d.tobytes())
