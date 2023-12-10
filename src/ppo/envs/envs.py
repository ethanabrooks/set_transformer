import os
from dataclasses import asdict, dataclass
from warnings import warn

import gymnasium as gym
import numpy as np
import pyglet
import torch
from gymnasium.spaces.box import Box as BoxSpace
from gymnasium.wrappers import OrderEnforcing, PassiveEnvChecker

from envs.base import Env
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


from miniworld.entity import Box  # noqa: E402
from miniworld.envs.oneroom import OneRoomS6Fast  # noqa: E402


class CustomOneRoomS6Fast(OneRoomS6Fast):
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
        info.update(state=self.state)
        return obs, info

    def step(self, action):
        info: dict
        obs, reward, done, truncated, info = super().step(action)
        info.update(state=self.state)
        return obs, reward, done, truncated, info


class BaseEnvWrapper(gym.Wrapper, Env):
    @property
    def action_space(self) -> gym.spaces.Discrete:
        return self.env.action_space

    @property
    def observation_space(self) -> Box:
        return self.env.observation_space


def make_env(env_name: str, seed: int, **kwargs):
    def _thunk():
        env: gym.Env = CustomOneRoomS6Fast(**kwargs)
        env = PassiveEnvChecker(env)
        env = OrderEnforcing(env)
        env = BaseEnvWrapper(env)

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
    envs = [
        make_env(env_name=env_name, seed=seed, **kwargs) for i in range(num_processes)
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

    def reset(self):
        obs, info = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs, info

    def step(self, action: torch.Tensor):
        action = action.detach().cpu().numpy()
        obs, reward, done, truncated, info = self.venv.step(action)
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).float()
        return obs, reward, done, truncated, info
