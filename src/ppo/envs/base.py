import gymnasium as gym
from gymnasium.spaces import Discrete


class Env(gym.Env):
    @property
    def task_space(self) -> Discrete:
        return None
