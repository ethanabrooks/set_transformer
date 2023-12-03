from abc import ABC, abstractmethod
from typing import Optional

import gymnasium
import torch
from gymnasium.spaces import Discrete, Space


class Env(gymnasium.Env, ABC):
    @property
    @abstractmethod
    def action_space(self) -> Discrete:
        pass

    @property
    @abstractmethod
    def observation_space(self) -> Space:
        pass

    @property
    def policy(self) -> Optional[torch.Tensor]:
        pass

    @property
    def values(self) -> Optional[torch.Tensor]:
        pass

    def optimal(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        pass
