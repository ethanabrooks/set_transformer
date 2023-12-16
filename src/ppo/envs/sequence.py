import numpy as np
from gymnasium.spaces import Discrete
from miniworld.entity import COLOR_NAMES, Ball, Box, Entity, Key
from miniworld.math import intersect_circle_segs

from envs.base import Env
from ppo.envs.one_room import OneRoom


class Sequence(OneRoom, Env):
    def __init__(
        self,
        *args,
        n_sequence: int,
        n_objects: int,
        rank: int,
        sequences: list[int],
        **kwargs,
    ):
        assert n_sequence >= 1
        assert n_objects >= n_sequence

        self.sequence = sequences[rank % len(sequences)][:n_sequence]
        print("rank:", rank, "sequence:", self.sequence)
        self.objects: list[Entity] = [
            obj_type(color=color)
            for obj_type in [Box, Ball, Key]
            for color in COLOR_NAMES
        ][:n_objects]
        super().__init__(*args, **kwargs, max_episode_steps=50 * n_sequence)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate(
            [
                *[obj.pos for obj in self.objects],
                self.agent.pos,
                self.agent.dir_vec,
                self.agent.right_vec,
            ],
            dtype=np.float32,
        )

    @property
    def task_space(self) -> Discrete:
        return Discrete(len(self.objects))

    def _gen_world(self):
        super()._gen_world()
        for obj in self.objects:
            self.place_entity(obj)

    def intersect(self, ent, pos, radius):  # noqa: Vulture
        """
        Check if an entity intersects with the world
        """

        # Ignore the Y position
        px, _, pz = pos
        pos = np.array([px, 0, pz])

        # Check for intersection with walls
        if intersect_circle_segs(pos, radius, self.wall_segs):
            return True

    def reset(self, *args, **kwargs):
        self.obj_iter = iter(self.sequence)
        self.target_obj_id = next(self.obj_iter)
        obs, info = super().reset(*args, **kwargs)
        info.update(task=self.target_obj_id)
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
        info.update(task=self.target_obj_id)
        return obs, reward, termination, truncated, info

    def calculate_optimal_action(self):  # noqa: Vulture
        # Vector from agent to target object
        target_pos = self.objects[self.target_obj_id].pos[[0, 2]]
        agent_pos = self.agent.pos[[0, 2]]
        print("target_pos", target_pos)
        print("agent pos", agent_pos)
        agent_to_target = target_pos - agent_pos
        print("distance", np.linalg.norm(agent_to_target))

        # Angle between agent's direction vector and agent-to-target vector
        dir_vec = self.agent.dir_vec[[0, 2]]

        angle = get_angle(dir_vec, agent_to_target)
        angle = np.degrees(angle)

        # Decide the action
        if abs(angle) < 45:
            # Move forward
            print(angle, "Move forward")
            return self.actions.move_forward
        else:
            # Turn left or right
            if angle < 0:
                print(angle, "Turn left")
                return self.actions.turn_left
            else:
                print(angle, "Turn right")
                return self.actions.turn_right


def get_angle(v1: np.ndarray, v2: np.ndarray):
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
