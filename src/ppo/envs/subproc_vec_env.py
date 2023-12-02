"""
Taken from https://github.com/openai/baselines
"""
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Callable, Optional

import gym
import numpy as np

from envs.base import Env


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class Command(Enum):
    ACTION_SPACE = auto()
    CLOSE = auto()
    OBSERVATION_SPACE = auto()
    RESET = auto()
    STEP = auto()


def work(env: Env, command: Command, data):
    if command == Command.ACTION_SPACE:
        return env.action_space
    elif command == Command.OBSERVATION_SPACE:
        return env.observation_space
    elif command == Command.RESET:
        return env.reset()
    elif command == Command.STEP:
        s, r, d, t, i = env.step(data)
        if d or t:
            s, _ = env.reset()
        return s, r, d, t, i
    raise RuntimeError(f"Unknown command {command}")


def worker(
    remote: Connection, parent_remote: Connection, env_fn_wrapper: CloudpickleWrapper
):
    parent_remote.close()
    env: Env = env_fn_wrapper.x()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == Command.CLOSE:
                break
            else:
                remote.send(work(env, cmd, data))
    finally:
        env.close()
        remote.close()


@dataclass
class SubprocVecEnv(gym.Env):
    n_processes: int
    closed: bool
    waiting: bool
    remotes: list[Connection]
    ps: list[Process]
    """
    VecEnv that runs multiple envs in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    @classmethod
    def make(cls, env_fns: Callable[[], Env]):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create envs to run in subprocesses. Need to be cloud-pickleable
        """
        remotes, ps = cls.start_processes(env_fns)
        return SubprocVecEnv(
            n_processes=len(env_fns),
            closed=False,
            waiting=False,
            remotes=remotes,
            ps=ps,
        )

    def _assert_not_closed(self):
        assert (
            not self.closed
        ), "Trying to operate on a SubprocVecEnv after calling close()"

    @property
    def action_space(self) -> gym.Space:
        return self.send_to_first(Command.ACTION_SPACE, None)

    @property
    def observation_space(self) -> gym.Space:
        return self.send_to_first(Command.OBSERVATION_SPACE, None)

    def close(self):
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send((Command.CLOSE, None))
        for p in self.ps:
            p.join()
        self.closed = True

    def reset(self, n: Optional[int] = None) -> np.ndarray:
        if n is None:
            observations, _ = zip(*self.send_to_all(Command.RESET, None))
            return np.stack(observations)
        else:
            return self.send_to_nth(n, Command.RESET, None)

    def send_to_all(self, command: Command, data):
        self._assert_not_closed()
        if data is None:
            for remote in self.remotes:
                remote.send((command, data))
        else:
            for remote, data in zip(self.remotes, data):
                remote.send((command, data))
        self.waiting = True
        received = [remote.recv() for remote in self.remotes]
        self.waiting = False
        return received

    def send_to_first(self, command: Command, data):
        return self.send_to_nth(0, command, data)

    def send_to_nth(self, n: int, command: Command, data):
        self._assert_not_closed()
        remote = self.remotes[n]
        remote.send((command, data))
        return remote.recv()

    @classmethod
    def start_processes(cls, env_fns) -> tuple[list, list[Process]]:
        remotes: list[Connection]
        work_remotes: list[Connection]
        remotes, work_remotes = zip(*[Pipe() for _ in range(len(env_fns))])
        ps = [
            Process(
                target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))
            )
            for (work_remote, remote, env_fn) in zip(work_remotes, remotes, env_fns)
        ]
        for p in ps:
            # if the main process crashes, we should not cause things to hang
            p.daemon = True
            p.start()
        for remote in work_remotes:
            remote.close()
        return remotes, ps

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        obs, rews, dones, truncated, infos = zip(
            *self.send_to_all(Command.STEP, actions)
        )
        return (
            np.stack(obs),
            np.stack(rews),
            np.stack(dones),
            np.stack(truncated),
            infos,
        )
