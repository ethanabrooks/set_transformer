from dataclasses import dataclass

import torch
from tensordict import TensorDict

from ppo.train import train
from sequence.base import Sequence as BaseSequence
from utils import Transition, filter_torchrl_warnings

filter_torchrl_warnings()

from torchrl.data import ReplayBuffer  # noqa: E402


@dataclass(frozen=True)
class Sequence(BaseSequence):
    @classmethod
    def make(
        cls,
        gamma: float,
        num_processes: int,
        num_steps: int,
        num_updates: int,
        trajectory_length: int,
        **kwargs,
    ):
        assert (num_processes * num_steps * num_updates) % trajectory_length == 0, (
            f"num_processes * num_steps * num_updates ({num_processes} * {num_steps} * {num_updates}) "
            f"must be divisible by sequence_length ({trajectory_length})"
        )
        replay_buffer: ReplayBuffer = train(
            gamma=gamma,
            **kwargs,
            load_path=None,
            num_processes=num_processes,
            num_steps=num_steps,
            num_updates=num_updates,
        )
        tensor_dict: TensorDict = replay_buffer[:]
        transitions = Transition(**tensor_dict.reshape(-1, trajectory_length))
        actions: torch.Tensor = transitions.actions
        pad_value = n_actions = 1 + actions.max().item()
        return Sequence(
            gamma=gamma,
            n_actions=n_actions,
            pad_value=pad_value,
            transitions=transitions,
        )
