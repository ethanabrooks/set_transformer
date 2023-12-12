import itertools

import numpy as np
from gymnasium.spaces.box import Box as BoxSpace
from miniworld.entity import COLOR_NAMES, Ball, Box, Entity, Key

from ppo.envs.one_room import OneRoom


class Sequence(OneRoom):
    def __init__(
        self,
        *args,
        n_sequence: int,
        n_objects: int,
        n_permutations: int,
        rank: int,
        **kwargs,
    ):
        assert n_sequence >= 1
        assert n_objects >= n_sequence
        self.objects: list[Entity] = [
            obj_type(color=color)
            for obj_type in [Box, Ball, Key]
            for color in COLOR_NAMES
        ][:n_objects]

        permutations = list(itertools.permutations(range(n_objects)))[:n_permutations]
        self.sequence = permutations[rank % len(permutations)][:n_sequence]
        self.eye = np.eye(len(self.objects))
        super().__init__(*args, **kwargs, max_episode_steps=50 * n_sequence)
        self.observation_space = BoxSpace(
            low=-np.inf, high=np.inf, shape=self.state.shape
        )

    @property
    def state(self) -> np.ndarray:
        return np.concatenate(
            [
                *[obj.pos for obj in self.objects],
                self.agent.pos,
                self.agent.dir_vec,
                self.agent.right_vec,
                self.eye[self.target_obj_id],
            ],
            dtype=np.float32,
        )

    def _gen_world(self):
        super()._gen_world()
        for obj in self.objects:
            self.place_entity(obj)

    def reset(self, *args, **kwargs):
        self.obj_iter = iter(self.sequence)
        self.target_obj_id = next(self.obj_iter)
        obs, info = super().reset(*args, **kwargs)
        obs = self.state
        return obs, info

    def step(self, action: np.ndarray):
        obs, _, _, truncated, info = super().step(action)
        reward = 0
        termination = False
        if self.near(self.objects[self.target_obj_id]):
            reward = 1
            try:
                self.target_obj_id = next(self.obj_iter)
            except StopIteration:
                termination = True
        obs = self.state
        return obs, reward, termination, truncated, info
