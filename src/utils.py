import math
import random
from dataclasses import dataclass
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

    def __len__(self):
        return len(self.states)


def decay_lr(lr: float, final_step: int, step: int, warmup_steps: int):
    if step < warmup_steps:
        # linear warmup
        lr_mult = float(step) / float(max(1, warmup_steps))
    else:
        # cosine learning rate decay
        progress = float(step - warmup_steps) / float(max(1, final_step - warmup_steps))
        progress = np.clip(progress, 0.0, 1.0)
        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr * lr_mult


def set_seed(seed: int):
    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # If you are using CUDA (GPU), you also need to set the seed for the CUDA device
    # This ensures reproducibility for GPU calculations as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for Python's random module
    random.seed(seed)
