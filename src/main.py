import datetime
import resource
import signal
import sys
import time
import urllib
from pathlib import Path
from typing import Optional, Set

import tomli
import wandb
from dollar_lambda import CommandTree, argument, option
from git import Repo
from ray import tune
from ray.air.integrations.wandb import setup_wandb
from rich import print
from wandb.sdk.wandb_run import Run

from evaluate import run
from param_space import param_space
from train.tabular import train as train_tabular_fn
from train.trajectories.train import train as train_trajectories_fn
from utils.main import get_config

tree = CommandTree()


def get_project_name():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["tool"]["poetry"]["name"]


def get_ignored() -> Set[Path]:
    repo = Repo(".")
    src = Path(__file__).parent
    ignored: str = repo.git.ls_files(str(src), o=True, i=True, exclude_standard=True)
    return set(Path(p).absolute() for p in ignored.splitlines())


get_ignored()


def check_dirty():
    assert not Repo(".").is_dirty()


def include_fn(path: str, exclude: list[str]):
    ignored = get_ignored()
    path = Path(path).absolute()
    include = path in ignored
    for pattern in exclude:
        include = include and not Path(path).match(pattern)
    return include


parsers = dict(config=option("config", default="tabular/cross-product"))


@tree.subcommand()
def evaluate(
    load_path: str,
    iterations: int,
    dummy_vec_env: bool = False,
    n_tokens: int = None,
    pad_value: int = None,
):
    config: dict = wandb.Api().run(load_path).config
    if "dummy_vec_env" not in config:
        config.update(dummy_vec_env=dummy_vec_env)
    config.update(load_path=load_path)
    if n_tokens is not None:
        config.update(n_tokens=n_tokens)
    if pad_value is not None:
        config.update(pad_value=pad_value)
    return run(**config, iterations=iterations, run=None)


@tree.subcommand(parsers=dict(name=argument("name"), **parsers))
def log(config: str, name: str, allow_dirty: bool = False, group: Optional[str] = None):
    if not allow_dirty:
        check_dirty()

    config_name = config
    config: dict = get_config(config)
    run = wandb.init(
        config=dict(**config, config=config_name),
        group=group,
        name=name,
        project=get_project_name(),
    )
    config.update(load_path=None)

    train(**config, run=run)


@tree.command(parsers=parsers)
def no_log(
    config: str,
    load_path: str = None,
    memory_limit: int = None,
    dummy_vec_env: bool = False,
):  # dead: disable
    def signal_handler(*_):
        print("Resource limit reached, terminating program.")
        sys.exit(1)

    # set resource limits
    if memory_limit is not None:
        # convert memory to bytes
        memory_bytes = memory_limit * (1024**2)

        # set memory limit
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

    # set signal handler for when resource limit is reached
    signal.signal(signal.SIGXCPU, signal_handler)

    if load_path is None:
        config: dict = get_config(config)
    else:
        config = wandb.Api().run(load_path).config

    config.update(load_path=load_path)
    if dummy_vec_env:
        config.update(dummy_vec_env=dummy_vec_env)
    return train(**config, run=None)


def get_git_rev():
    repo = Repo(".")
    if repo.head.is_detached:
        return repo.head.object.name_rev
    else:
        return repo.active_branch.commit.name_rev


@tree.subcommand(parsers=parsers)
def sweep(
    config: str,
    gpus_per_proc: int,
    group: str = None,
    notes: str = None,
    num_samples: int = None,
    allow_dirty: bool = False,
):
    if group is None:
        group = datetime.datetime.now().strftime("-%d-%m-%H:%M:%S")
    commit = get_git_rev()
    project_name = get_project_name()
    config_name = config
    config = get_config(config)
    if not allow_dirty:
        check_dirty()

    def trainable(sweep_params: dict):
        sleep_time = 1
        k: str
        for k, v in sweep_params.items():
            *path, key = k.split("/")
            subconfig = config

            for name in path:
                subconfig = subconfig[name]
            if key not in subconfig:
                print(config)
                raise ValueError(f"Failed to index into config with path {k}")
            subconfig[key] = v

        run: Run
        while True:
            try:
                run = setup_wandb(
                    config=dict(**config, config_name=config_name, commit=commit),
                    group=group,
                    project=project_name,
                    rank_zero_only=False,
                    notes=notes,
                    resume="never",
                )
                break
            except wandb.errors.CommError:
                time.sleep(sleep_time)
                sleep_time *= 2
        print(
            f"wandb: ️👪 View group at {run.get_project_url()}/groups/{urllib.parse.quote(group)}/workspace"
        )
        root = Path(__file__).parent
        run.log_code(
            str(root), include_fn=lambda p: include_fn(p, exclude=["__pycache__/**"])
        )  # log untracked files
        config.update(run=run, load_path=None)
        return train(**config)

    tune.Tuner(
        trainable=tune.with_resources(trainable, dict(gpu=gpus_per_proc)),
        tune_config=None if num_samples is None else dict(num_samples=num_samples),
        param_space=param_space,
    ).fit()


def train(*args, train_trajectories: bool, dummy_vec_env: bool = False, **kwargs):
    return (
        train_trajectories_fn(*args, dummy_vec_env=dummy_vec_env, **kwargs)
        if train_trajectories
        else train_tabular_fn(*args, **kwargs)
    )


if __name__ == "__main__":
    tree()
