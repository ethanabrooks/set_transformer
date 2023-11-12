from dataclasses import dataclass
from enum import Enum, auto
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


class SampleFrom(Enum):
    TRAJECTORIES = auto()
    UNIFORM = auto()


def tensor_hash(tensor: torch.Tensor):
    # Ensure that the tensor is on the CPU and converted to 1D
    tensor_1d: torch.Tensor = tensor.cpu().flatten()
    array_1d: np.ndarray = tensor_1d.numpy()

    # Convert the 1D tensor to bytes and hash
    return hash(array_1d.tobytes())
