import os
from enum import Enum, auto
from warnings import warn

import gymnasium as gym
import numpy as np
import pyglet
import torch
from gymnasium.spaces.box import Box
from gymnasium.wrappers import OrderEnforcing, PassiveEnvChecker

from envs.base import Env
from ppo.envs.base import Env as PPOEnv
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


from ppo.envs.one_room import OneRoom  # noqa: E402
from ppo.envs.pickup import Pickup  # noqa: E402
from ppo.envs.sequence import Sequence  # noqa: E402


class EnvType(Enum):
    CHEETAH = auto()
    ONE_ROOM = auto()
    PICKUP = auto()
    SEQUENCE = auto()


class BaseEnvWrapper(gym.Wrapper, PPOEnv, Env):
    def __init__(self, env: PPOEnv, n_tasks: int, rank: int):
        super().__init__(env)
        self.n_tasks = n_tasks
        self.rank = rank

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.Space:
        return self.env.observation_space

    @property
    def task_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.n_tasks)

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        info.update(task=self.rank)
        return obs, info

    def step(self, action):
        info: dict
        obs, reward, done, truncated, info = super().step(action)
        info.update(task=self.rank)
        return obs, reward, done, truncated, info


def make_env(env_type: EnvType, n_tasks: int, rank: int, seed: int, **kwargs):
    def _thunk():
        if env_type == EnvType.ONE_ROOM:
            env: gym.Env = OneRoom(**kwargs)
        elif env_type == EnvType.CHEETAH:
            env: gym.Env = gym.make("HalfCheetah-v2")
        elif env_type == EnvType.PICKUP:
            env: gym.Env = Pickup(**kwargs)
        elif env_type == EnvType.SEQUENCE:
            env: gym.Env = Sequence(**kwargs, rank=rank)
        else:
            raise ValueError(f"Unknown env_type: {env_type}")
        env = PassiveEnvChecker(env)
        env = OrderEnforcing(env)
        env = BaseEnvWrapper(env, n_tasks=n_tasks, rank=rank)

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
    kwargs.update(n_tasks=num_processes)  # TODO: Allow n_tasks < num_processes
    envs = [make_env(rank=i, env_type=env_type, **kwargs) for i in range(num_processes)]

    envs: SubprocVecEnv = (
        DummyVecEnv.make(envs)
        if dummy_vec_env or len(envs) == 1
        else SubprocVecEnv.make(envs)
    )

    if env_type == EnvType.CHEETAH:
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

    @property
    def task_space(self):
        return self.venv.task_space
