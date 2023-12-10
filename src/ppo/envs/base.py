from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


class Env(gym.Env, ABC):
    @property
    @abstractmethod
    def action_space(self) -> Discrete:
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self) -> Box:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        raise NotImplementedError
