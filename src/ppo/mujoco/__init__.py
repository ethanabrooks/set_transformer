# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from .ant import AntEnv
from .half_cheetah import HalfCheetahEnv
from .hopper import HopperEnv
from .humanoid import HumanoidEnv
from .humanoidstandup import HumanoidStandupEnv
from .inverted_double_pendulum import InvertedDoublePendulumEnv
from .inverted_pendulum import InvertedPendulumEnv
from .mujoco_env import MujocoEnv
from .pusher import PusherEnv
from .reacher import ReacherEnv
from .striker import StrikerEnv
from .swimmer import SwimmerEnv
from .thrower import ThrowerEnv
from .walker2d import Walker2dEnv

# whitelist
AntEnv
HalfCheetahEnv
HopperEnv
HumanoidEnv
HumanoidStandupEnv
InvertedDoublePendulumEnv
InvertedPendulumEnv
MujocoEnv
PusherEnv
ReacherEnv
StrikerEnv
SwimmerEnv
ThrowerEnv
Walker2dEnv
