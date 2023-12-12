import numpy as np
from miniworld.entity import Box
from miniworld.envs.oneroom import OneRoomS6Fast


class OneRoom(OneRoomS6Fast):
    @property
    def state(self):
        box: Box = self.box
        return np.concatenate(
            [
                box.pos,
                self.agent.pos,
                self.agent.dir_vec,
                self.agent.right_vec,
            ]
        )

    def reset(self, *args, **kwargs):
        obs, info = super().reset(*args, **kwargs)
        info.update(state=self.state)
        return obs, info

    def step(self, action):
        info: dict
        obs, reward, done, truncated, info = super().step(action)
        info.update(state=self.state)
        return obs, reward, done, truncated, info
