from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from gymnasium.spaces import Space

from utils.dataclasses import Transition


@dataclass
class DataStorage:
    action_space: Space
    memmap: np.memmap
    observation_space: Space

    @staticmethod
    def make(
        action_space: Space,
        action_probs_shape: tuple[int, ...],
        num_processes,
        num_timesteps: int,
        obs_space: Space,
        path: Path,
        state_shape: tuple[int, ...],
    ):
        *action_shape, _ = action_probs_shape
        obs_type = (np.float32, obs_space.shape)
        state_type = (np.float32, state_shape)

        transition_type = Transition[tuple[int, ...]](
            states=state_type,
            actions=(action_space.dtype, tuple(action_shape)),
            action_probs=(np.float32, action_probs_shape),
            rewards=(np.float32, ()),
            done=(bool, ()),
            next_states=state_type,
            obs=obs_type,
            next_obs=obs_type,
        )
        dtype = np.dtype(
            [(k, dt, shape) for k, (dt, shape) in asdict(transition_type).items()]
        )
        return DataStorage(
            action_space=action_space,
            memmap=np.memmap(
                str(path),
                dtype=dtype,
                mode="w+",
                shape=(num_timesteps, num_processes),
            ),
            observation_space=obs_space,
        )

    @staticmethod
    def make_path(directory: Path):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        directory = directory / now
        directory.mkdir(exist_ok=True, parents=True)
        return directory / "replay-buffer.dat"

    def insert(self, timestep: int, transition: Transition):
        dtype: np.dtypes.VoidDType = self.memmap.dtype
        transition = asdict(transition)
        for field in dtype.names:
            self.memmap[field][timestep] = transition[field]

    def to_transition(self) -> Transition:
        dtype: np.dtypes.VoidDType = self.memmap.dtype
        return Transition(**{k: self.memmap[k] for k in dtype.names})
