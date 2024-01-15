from dataclasses import asdict, dataclass

import numpy as np
import torch
from gymnasium.spaces import Discrete

from ppo.data_storage import DataStorage
from ppo.train import train
from sequence.base import Sequence as BaseSequence
from utils import Transition


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
        data_storage: DataStorage = train(
            gamma=gamma,
            **kwargs,
            load_path=None,
            num_processes=num_processes,
            num_steps=num_steps,
            num_updates=num_updates,
        )

        def preprocess():
            v: np.ndarray
            for k, v in asdict(data_storage.to_transition()).items():
                v = v.swapaxes(0, 1)
                _, _, *shape = v.shape
                reshape = v.reshape(-1, trajectory_length, *shape)
                tensor = torch.from_numpy(reshape)
                yield k, tensor

        transitions = Transition(**dict(preprocess()))
        assert isinstance(data_storage.action_space, Discrete)
        pad_value = n_actions = data_storage.action_space.n
        if isinstance(data_storage.observation_space, Discrete):
            pad_value += data_storage.observation_space.n

        return Sequence(
            gamma=gamma,
            n_actions=n_actions,
            pad_value=pad_value,
            transitions=transitions,
        )
