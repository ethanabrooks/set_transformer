from copy import deepcopy
from typing import Optional

import numpy as np
from gymnasium.spaces import Discrete
from miniworld.entity import Box
from miniworld.envs.maze import MazeS3Fast

from envs.base import Env

TEXTURES = [
    "brick_wall",
    "cinder_blocks",
    "drywall",
    "grass",
    "lava",
    "rock",
    "slime",
    "water",
    "wood",
    "cardboard",
]


class Maze(MazeS3Fast, Env):
    def __init__(self, num_processes: Optional[int], rank: int, seed: int, **kwargs):
        self.rank = rank
        self.seed = rank + seed
        self.num_processes = num_processes
        super().__init__(**kwargs)

    @property
    def task_space(self) -> Discrete:
        if self.num_processes is not None:
            return Discrete(self.num_processes)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        #     import matplotlib.pyplot as plt

        #     print("SEED:", self.seed)

        #     plt.imsave(f"{self.seed}.png", self.render_top_view())
        #     breakpoint()
        info.update(task=self.rank)
        return obs, info

    def step(self, action):
        info: dict
        *res, info = super().step(action)
        info.update(task=self.rank)
        return *res, info

    def __gen_world(self):
        rows = []

        # For each row
        for j in range(self.num_rows):
            row = []

            # For each column
            for i in range(self.num_cols):
                min_x = i * (self.room_size + self.gap_size)
                max_x = min_x + self.room_size

                min_z = j * (self.room_size + self.gap_size)
                max_z = min_z + self.room_size

                room = self.add_rect_room(
                    min_x=min_x,
                    max_x=max_x,
                    min_z=min_z,
                    max_z=max_z,
                    wall_tex=TEXTURES[(j * self.num_cols + i) % len(TEXTURES)]
                    # floor_tex='asphalt'
                )
                row.append(room)

            rows.append(row)

        visited = set()

        def visit(i, j):
            """
            Recursive backtracking maze construction algorithm
            https://stackoverflow.com/questions/38502
            """

            room = rows[j][i]

            visited.add(room)

            # Reorder the neighbors to visit in a random order
            orders = [(0, 1), (0, -1), (-1, 0), (1, 0)]
            assert 4 <= len(orders)
            neighbors = []

            while len(neighbors) < 4:
                elem = orders[self.np_random.choice(len(orders))]
                orders.remove(elem)
                neighbors.append(elem)

            # For each possible neighbor
            for dj, di in neighbors:
                ni = i + di
                nj = j + dj

                if nj < 0 or nj >= self.num_rows:
                    continue
                if ni < 0 or ni >= self.num_cols:
                    continue

                neighbor = rows[nj][ni]

                if neighbor in visited:
                    continue

                if di == 0:
                    self.connect_rooms(
                        room, neighbor, min_x=room.min_x, max_x=room.max_x
                    )
                elif dj == 0:
                    self.connect_rooms(
                        room, neighbor, min_z=room.min_z, max_z=room.max_z
                    )

                visit(ni, nj)

        # Generate the maze starting from the top-left corner
        visit(0, 0)

        self.box = self.place_entity(Box(color="red"))

        self.place_agent()

    def _gen_world(self):
        np_random = deepcopy(self.np_random)
        self.np_random = np.random.RandomState(self.seed)
        self.__gen_world()
        self.np_random = np_random
        self.place_agent()

    def _reward(self):  # noqa: Vulture
        return 1
