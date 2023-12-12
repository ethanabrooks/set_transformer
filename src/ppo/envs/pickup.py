import itertools

from gymnasium import spaces, utils
from miniworld.entity import COLOR_NAMES, Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv


class IdMixin:
    def __init__(self, id: int, **kwargs) -> None:
        self.id = id
        super().__init__(**kwargs)


class Ball(IdMixin, Ball):
    pass


class Box(IdMixin, Box):
    pass


class Key(IdMixin, Key):
    pass


class Pickup(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move_back                   |
    | 4   | pickup                      |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +1 when agent picked up object

    ## Arguments

    ```python
    PickupObjects(size=12, num_objs=5)
    ```

    `size`: size of world

    `num_objs`: number of objects

    """

    def __init__(self, size=12, num_objs=5, **kwargs):
        assert size >= 2
        self.size = size
        self.num_objs = num_objs

        MiniWorldEnv.__init__(self, max_episode_steps=400, **kwargs)
        utils.EzPickle.__init__(self, size, num_objs, **kwargs)

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.pickup + 1)

    def _gen_world(self):
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        obj_types = [Ball, Box, Key]
        colorlist = list(COLOR_NAMES)

        def generate():
            for i, obj in enumerate(range(self.num_objs)):
                pairs = itertools.product(obj_types, colorlist)
                for obj_type, color in itertools.islice(pairs, self.num_objs):
                    if obj_type == Key:
                        yield obj_type(color=color, id=i)
                    else:
                        yield obj_type(color=color, size=0.9, id=i)

        self.objects = list(generate())
        for object in self.objects:
            self.place_entity(object)
            self.target_obj = object

        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.agent.carrying:
            self.entities.remove(self.agent.carrying)
            self.agent.carrying = None
            reward = 1

            termination = True

        return obs, reward, termination, truncation, info
