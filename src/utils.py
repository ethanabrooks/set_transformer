from dataclasses import dataclass
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
