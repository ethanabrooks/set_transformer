import itertools
import os
from enum import Enum, auto
from warnings import warn

import gymnasium as gym
import numpy as np
import pyglet
import torch
from gymnasium.spaces.box import Box as BoxSpace
from gymnasium.wrappers import OrderEnforcing, PassiveEnvChecker

from envs.base import Env
from ppo.envs.base import Env as PPOEnv
from ppo.envs.dummy_vec_env import DummyVecEnv
from ppo.envs.monitor import Monitor
from ppo.envs.subproc_vec_env import SubprocVecEnv

pyglet.options["headless"] = True
try:
    headless_device = int(os.environ["CUDA_VISIBLE_DEVICES"])
except (KeyError, ValueError):
    warn("CUDA_VISIBLE_DEVICES not set, defaulting to 0.")
    headless_device = 0
pyglet.options["headless_device"] = headless_device


from miniworld.entity import COLOR_NAMES, Ball, Box, Key  # noqa: E402
from miniworld.envs.oneroom import OneRoomS6Fast  # noqa: E402


class CustomOneRoomS6Fast(OneRoomS6Fast):
    def __init__(self, *args, rank: int, **kwargs):
        self.rank = rank
        super().__init__(*args, **kwargs)

    @property
    def state(self):
        box: Box = self.box
        return np.concatenate(
            [
                box.pos,
                self.agent.pos,
                self.agent.dir_vec,
                self.agent.right_vec,
            ]
        )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        info.update(state=self.state, task=self.rank)
        return obs, info

    def step(self, action):
        info: dict
        obs, reward, done, truncated, info = super().step(action)
        info.update(state=self.state, task=self.rank)
        return obs, reward, done, truncated, info


class SequenceEnv(CustomOneRoomS6Fast):
    def __init__(self, *args, n_sequence: int, n_objects: int, rank: int, **kwargs):
        assert n_sequence >= 1
        assert n_objects >= n_sequence
        self.objects = [
            obj_type(color=color)
            for obj_type in [Box, Ball, Key]
            for color in COLOR_NAMES
        ][:n_objects]

        permutations = list(itertools.permutations(self.objects))
        self.sequence = permutations[rank % len(permutations)][:n_sequence]
        super().__init__(*args, **kwargs, max_episode_steps=50 * n_sequence, rank=rank)

    def _gen_world(self):
        super()._gen_world()
        for obj in self.objects:
            self.place_entity(obj)

    def reset(self, *args, **kwargs):
        self.obj_iter = iter(self.sequence)
        self.target_obj = next(self.obj_iter)
        obs, info = super().reset(*args, **kwargs)
        return obs, info

    def step(self, action: np.ndarray):
        obs, _, _, truncated, info = super().step(action)
        reward = 0
        termination = False
        if self.near(self.target_obj):
            reward += self._reward()
            try:
                self.target_obj = next(self.obj_iter)
            except StopIteration:
                termination = True
        return obs, reward, termination, truncated, info


class EnvType(Enum):
    ONE_ROOM = auto()
    SEQUENCE = auto()


class BaseEnvWrapper(gym.Wrapper, PPOEnv, Env):
    def __init__(self, env: PPOEnv, n_tasks: int):
        super().__init__(env)
        self.n_tasks = n_tasks

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.env.action_space

    @property
    def observation_space(self) -> Box:
        return self.env.observation_space

    @property
    def task_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(self.n_tasks)


def make_env(env_name: str, n_tasks: int, seed: int, **kwargs):
    env_type = EnvType[env_name]

    def _thunk():
        if env_type == EnvType.ONE_ROOM:
            env: gym.Env = CustomOneRoomS6Fast(**kwargs)
        elif env_type == EnvType.SEQUENCE:
            env: gym.Env = SequenceEnv(**kwargs)
        else:
            raise ValueError(f"Unknown env_type: {env_type}")
        env = PassiveEnvChecker(env)
        env = OrderEnforcing(env)
        env = BaseEnvWrapper(env, n_tasks=n_tasks)

        env = Monitor(env=env, filename=None, allow_early_resets=True)
        env.reset(seed=seed)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(
    env_name: str,
    seed: int,
    num_processes: int,
    gamma: float,
    device: torch.device,
    dummy_vec_env: bool,
    **kwargs,
) -> "VecPyTorch":
    kwargs.update(n_tasks=num_processes)  # TODO: Allow n_tasks < num_processes
    envs = [
        make_env(env_name=env_name, rank=i, seed=seed, **kwargs)
        for i in range(num_processes)
    ]

    envs: SubprocVecEnv = (
        DummyVecEnv.make(envs)
        if dummy_vec_env or len(envs) == 1
        else SubprocVecEnv.make(envs)
    )

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
        assert isinstance(self.observation_space, BoxSpace)
        obs_shape = self.observation_space.shape
        self.observation_space = BoxSpace(
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
