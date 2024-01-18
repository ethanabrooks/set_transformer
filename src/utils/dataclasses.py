from dataclasses import dataclass
from typing import Generic, NamedTuple, Optional, TypeVar

import torch

T = TypeVar("T")


class DataPoint(NamedTuple):
    action_probs: torch.Tensor
    actions: torch.Tensor
    done: torch.Tensor
    idx: torch.Tensor
    input_q: torch.Tensor
    n_bellman: torch.Tensor
    next_obs: torch.Tensor
    obs: torch.Tensor
    rewards: torch.Tensor
    target_q: torch.Tensor


@dataclass(frozen=True)
class Transition(Generic[T]):
    states: T
    actions: T
    action_probs: T
    next_states: T
    rewards: T
    done: T
    obs: Optional[T] = None
    next_obs: Optional[T] = None

    def __getitem__(self, idx: torch.Tensor):
        def index(x: Optional[torch.Tensor]):
            if isinstance(idx, torch.Tensor):
                x = x.to(idx.device)
            if isinstance(x, torch.Tensor):
                return x[idx]
            return x

        return type(self)(
            states=index(self.states),
            obs=index(self.obs),
            actions=index(self.actions),
            action_probs=index(self.action_probs),
            next_states=index(self.next_states),
            next_obs=index(self.next_obs),
            rewards=index(self.rewards),
            done=index(self.done),
        )

    def __len__(self):
        return len(self.states)
