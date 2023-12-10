from abc import ABC
from typing import Optional

import gymnasium
import torch
from gymnasium.spaces import Discrete


class Env(gymnasium.Env, ABC):
    @property
    def policy(self) -> Optional[torch.Tensor]:
        pass

    @property
    def task_space(self) -> Optional[Discrete]:
        pass

    @property
    def values(self) -> Optional[torch.Tensor]:
        pass

    def optimal(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        pass
