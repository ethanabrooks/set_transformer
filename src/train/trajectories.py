from typing import Optional

from wandb.sdk.wandb_run import Run

from envs.dummy_vec_env import DummyVecEnv
from envs.subproc_vec_env import SubprocVecEnv
from grid_world.base import GridWorld
from grid_world.env import Env
from grid_world.values import GridWorldWithValues
from sequence import make_grid_world_sequence
from train.trainer.base import Trainer
from utils import set_seed


def train(
    *args,
    dummy_vec_env: bool,
    grid_world_args: dict,
    partial_observation: bool,
    rmse_bellman: float,
    run: Run,
    seed: int,
    sequence_args: dict,
    test_size: int,
    time_limit: int,
    train_size: int,
    config: Optional[str] = None,
    **kwargs,
):
    del config
    set_seed(seed)

    def make_grid_world(n_tasks: int, seed: int):
        return GridWorld.make(
            **grid_world_args, n_tasks=n_tasks, seed=seed, terminal_transitions=None
        )

    sequence = make_grid_world_sequence(
        grid_world=make_grid_world(n_tasks=train_size, seed=seed),
        partial_observation=partial_observation,
        sample_from_trajectories=True,
        **sequence_args,
        stop_at_rmse=rmse_bellman,
        time_limit=time_limit,
    )

    def make_env(i: int):
        return lambda: Env(
            grid_world=GridWorldWithValues.make(
                stop_at_rmse=rmse_bellman,
                grid_world=make_grid_world(n_tasks=1, seed=seed + i),
            ),
            time_limit=time_limit,
        )

    env_fns = list(map(make_env, range(test_size)))
    envs = DummyVecEnv.make(env_fns) if dummy_vec_env else SubprocVecEnv.make(env_fns)
    return Trainer().compute_values(
        *args,
        envs=envs,
        **kwargs,
        partial_observation=partial_observation,
        rmse_bellman=rmse_bellman,
        run=run,
        sequence=sequence,
    )
