"""
Taken from https://github.com/openai/baselines
"""
from dataclasses import dataclass
from enum import Enum, auto
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Optional

import numpy as np
import torch
from gym.spaces import Discrete

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
    OPTIMAL = auto()
    POLICY = auto()
    RESET = auto()
    STEP = auto()
    VALUES = auto()


def work(env: Env, command: Command, data):
    if command == Command.ACTION_SPACE:
        return env.action_space
    elif command == Command.OBSERVATION_SPACE:
        return env.observation_space
    elif command == Command.RESET:
        return env.reset()
    elif command == Command.OPTIMAL:
        return env.optimal(data)
    elif command == Command.POLICY:
        return env.policy
    elif command == Command.STEP:
        return env.step(data)
    elif command == Command.VALUES:
        return env.values
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
class SubprocVecEnv:
    remotes: list[Connection]
    ps: list[Process]
    closed: bool
    waiting: bool

    """
    VecEnv that runs multiple envs in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    @classmethod
    def make(cls, env_fns):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create envs to run in subprocesses. Need to be cloud-pickleable
        """
        remotes: list[Connection]
        remotes, ps = cls.start_processes(env_fns)
        return cls(remotes, ps, closed=False, waiting=False)

    def _assert_not_closed(self):
        assert (
            not self.closed
        ), "Trying to operate on a SubprocVecEnv after calling close()"

    @property
    def action_space(self) -> Discrete:
        return self.send_to_first(Command.ACTION_SPACE, None)

    @property
    def observation_space(self):
        return self.send_to_first(Command.OBSERVATION_SPACE, None)

    @property
    def policy(self):
        return self.collect_optional(Command.POLICY)

    @property
    def values(self):
        return self.collect_optional(Command.VALUES)

    def close(self):
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send((Command.CLOSE, None))
        for p in self.ps:
            p.join()
        self.closed = True

    def collect_optional(self, command: Command):
        xs = [x for x in self.send_to_all(command, None) if x is not None]
        min_size = min(len(x) for x in xs)
        if xs:
            return torch.stack([x[:min_size] for x in xs])

    def optimal(self, n: int, state: torch.Tensor) -> Optional[float]:
        return self.send_to_nth(n, Command.OPTIMAL, state)

    def reset(self, n: Optional[int] = None) -> np.ndarray:
        if n is None:
            return np.stack(self.send_to_all(Command.RESET, None))
        else:
            return self.send_to_nth(n, Command.RESET, None)

    def send_to_all(self, command: Command, data):
        self._assert_not_closed()
        if data is None:
            for remote in self.remotes:
                remote.send((command, data))
        else:
            assert len(self.remotes) == len(data)
            for i, (remote, data) in enumerate(zip(self.remotes, data)):
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
        obs, rews, dones, infos = zip(*self.send_to_all(Command.STEP, actions))
        return np.stack(obs), np.stack(rews), np.stack(dones), infos
