from typing import Optional

from wandb.sdk.wandb_run import Run

from envs.dummy_vec_env import DummyVecEnv
from envs.subproc_vec_env import SubprocVecEnv
from grid_world.base import GridWorld
from grid_world.env import Env
from grid_world.values import GridWorldWithValues
from ppo.envs.envs import make_env as make_ppo_env
from sequence import make_grid_world_sequence
from sequence.ppo import Sequence as PPOSequence
from train.trainer.base import Trainer
from train.trainer.grid_world import Trainer as GridWorldTrainer
from train.trainer.miniworld import Trainer as MiniWorldTrainer
from utils import set_seed


def make_grid_world_sequence_and_env_fn(
    grid_world_args: dict,
    partial_observation: bool,
    rmse_bellman: float,
    seed: int,
    time_limit: int,
    train_size: int,
    **kwargs: dict,
):
    def make_grid_world(n_tasks: int, seed: int):
        return GridWorld.make(
            **grid_world_args, n_tasks=n_tasks, seed=seed, terminal_transitions=None
        )

    sequence = make_grid_world_sequence(
        **kwargs,
        grid_world=make_grid_world(n_tasks=train_size, seed=seed),
        partial_observation=partial_observation,
        sample_from_trajectories=True,
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

    return sequence, make_env


def make_ppo_sequence_and_env_fn(env_name: str, seed: int, **sequence_args):
    sequence = PPOSequence.make(**sequence_args, env_name=env_name, seed=seed, run=None)

    def make_env(i: int):
        return make_ppo_env(env_id=env_name, seed=seed + i)

    return sequence, make_env


def train(
    dummy_vec_env: bool,
    evaluator_args: dict,
    load_path: Optional[str],
    lr: float,
    model_args: dict,
    n_plot: dict,
    rmse_bellman: float,
    run: Run,
    seed: int,
    test_size: int,
    train_args: dict,
    use_grid_world: bool,
    config: Optional[str] = None,
    **kwargs,
):
    del config
    set_seed(seed)

    if use_grid_world:
        sequence, env_fn = make_grid_world_sequence_and_env_fn(
            **kwargs, rmse_bellman=rmse_bellman, seed=seed
        )

    else:
        sequence, env_fn = make_ppo_sequence_and_env_fn(
            dummy_vec_env=dummy_vec_env, lr=lr, **kwargs, seed=seed
        )

    env_fns = list(map(env_fn, range(test_size)))
    envs = DummyVecEnv.make(env_fns) if dummy_vec_env else SubprocVecEnv.make(env_fns)
    if use_grid_world:
        trainer: Trainer = GridWorldTrainer.make(
            envs=envs,
            evaluator_args=evaluator_args,
            load_path=load_path,
            lr=lr,
            model_args=model_args,
            n_plot=n_plot,
            rmse_bellman=rmse_bellman,
            run=run,
            sequence=sequence,
            **train_args,
        )
    else:
        trainer: Trainer = MiniWorldTrainer.make(
            envs=envs,
            evaluator_args=evaluator_args,
            load_path=load_path,
            lr=lr,
            model_args=model_args,
            n_plot=n_plot,
            rmse_bellman=rmse_bellman,
            run=run,
            sequence=sequence,
            **train_args,
        )
    return trainer.train(lr=lr)
