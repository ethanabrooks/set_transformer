from abc import ABC, abstractmethod
from typing import Optional

import gym
import gym.spaces
import torch


class Env(gym.Env, ABC):
    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Discrete:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> gym.Space:
        pass

    @property
    @abstractmethod
    def policy(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def values(self) -> torch.Tensor:
        pass

    def optimal(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        pass
