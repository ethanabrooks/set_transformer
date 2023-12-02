from abc import ABC, abstractmethod

import gym
import numpy as np


class Env(gym.Env, ABC):
    @property
    @abstractmethod
    def action_space(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def observation_space(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        raise NotImplementedError
