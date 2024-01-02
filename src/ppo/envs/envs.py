import itertools
import os
from enum import Enum, auto
from functools import lru_cache
from typing import Optional
from warnings import warn

import gymnasium as gym
import numpy as np
import pyglet
import torch
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import OrderEnforcing, PassiveEnvChecker

from ppo.envs.dummy_vec_env import DummyVecEnv
from ppo.envs.monitor import Monitor
from ppo.envs.subproc_vec_env import SubprocVecEnv
from ppo.envs.vec_normalize import VecNormalize

pyglet.options["headless"] = True
try:
    headless_device = int(os.environ["CUDA_VISIBLE_DEVICES"])
except (KeyError, ValueError):
    warn("CUDA_VISIBLE_DEVICES not set, defaulting to 0.")
    headless_device = 0
pyglet.options["headless_device"] = headless_device


from ppo.envs.maze import Maze  # noqa: E402
from ppo.envs.one_room import OneRoom  # noqa: E402
from ppo.envs.sequence import Sequence  # noqa: E402


class EnvType(Enum):
    CHEETAH = auto()
    GRID_WORLD = auto()  # noqa: Vulture
    MAZE = auto()
    ONE_ROOM = auto()
    SEQUENCE = auto()


@lru_cache
def all_sequences(n_sequence: int, n_objects: int) -> np.ndarray:
    sequences = np.array(list(itertools.permutations(range(n_objects), n_sequence)))
    perm = np.random.permutation(len(sequences))
    return sequences[perm]


def get_sequences(
    n_permutations: int, permutation_starting_idx: int, **kwargs
) -> np.ndarray:
    sequences = all_sequences(**kwargs)
    return sequences[
        permutation_starting_idx : permutation_starting_idx + n_permutations
    ]


def make_env(
    env_type: EnvType,
    rank: int,
    seed: int,
    num_processes: Optional[int] = None,
    **kwargs,
):
    if env_type == EnvType.SEQUENCE:
        sequences = get_sequences(**kwargs)
        kwargs.update(sequences=sequences)
        del kwargs["n_permutations"]
        del kwargs["permutation_starting_idx"]

    def _thunk():
        if env_type == EnvType.ONE_ROOM:
            env: gym.Env = OneRoom(**kwargs)
        elif env_type == EnvType.CHEETAH:
            env: gym.Env = gym.make("HalfCheetah-v2")
        elif env_type == EnvType.MAZE:
            env: gym.Env = Maze(
                **kwargs, num_processes=num_processes, rank=rank, seed=rank + seed
            )
        elif env_type == EnvType.SEQUENCE:
            env: gym.Env = Sequence(**kwargs, rank=rank)
        else:
            raise ValueError(f"Unknown env_type: {env_type}")
        env = PassiveEnvChecker(env)
        env = OrderEnforcing(env)

        env = Monitor(env=env, filename=None, allow_early_resets=True)
        env.reset(seed=seed)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(
    env_type: EnvType,
    num_processes: int,
    gamma: float,
    device: torch.device,
    dummy_vec_env: bool,
    **kwargs,
) -> "VecPyTorch":
    envs = [
        make_env(env_type=env_type, num_processes=num_processes, rank=i, **kwargs)
        for i in range(num_processes)
    ]

    envs: SubprocVecEnv = (
        DummyVecEnv.make(envs)
        if dummy_vec_env or len(envs) == 1
        else SubprocVecEnv.make(envs)
    )

    if len(envs.observation_space.shape) == 1:
        if gamma is None:
            envs = VecNormalize(envs, ret=False)
        else:
            envs = VecNormalize(envs, gamma=gamma)

    envs = VecPyTorch(envs, device)

    return envs


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env: gym.Env = None, op: list[int] = [2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        assert isinstance(self.observation_space, Box)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype,
        )

    def observation(self, ob: torch.Tensor):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(gym.Wrapper):
    def __init__(self, venv: SubprocVecEnv, device: torch.Tensor):
        """Return only every `skip`-th frame"""
        super().__init__(venv)
        self.venv = venv
        self.device = device
        # TODO: Fix data types

    @property
    def task_space(self) -> Discrete:
        return self.venv.task_space

    def obs_to_tensor(self, obs: np.ndarray) -> list[torch.Tensor]:
        return torch.from_numpy(obs).float().to(self.device)

    def reset(self):
        obs: np.ndarray
        obs, info = self.venv.reset()
        return self.obs_to_tensor(obs), info

    def step(self, action: torch.Tensor):
        action = action.detach().cpu().numpy()
        obs, reward, done, truncated, info = self.venv.step(action)
        obs = self.obs_to_tensor(obs)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, truncated, info
