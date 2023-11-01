from dataclasses import dataclass
import torch
from typing import Generic, TypeVar

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
