from miniworld.envs.maze import MazeS3Fast

from envs.base import Env


class Maze(MazeS3Fast, Env):
    def _reward(self):  # noqa: Vulture
        return 1
