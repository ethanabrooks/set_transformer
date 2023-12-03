from abc import ABC
from typing import Optional

import gymnasium
import torch


class Env(gymnasium.Env, ABC):
    @property
    def policy(self) -> Optional[torch.Tensor]:
        pass

    @property
    def values(self) -> Optional[torch.Tensor]:
        pass

    def optimal(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        pass
