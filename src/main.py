import datetime
import resource
import signal
import sys
import time
import urllib
from pathlib import Path
from typing import Set

import tomli
from dollar_lambda import CommandTree, argument, option
from git import Repo
from omegaconf import DictConfig, OmegaConf
from ray import tune
from ray.air.integrations.wandb import setup_wandb
from rich import print
from wandb.sdk.wandb_run import Run

import wandb
from param_space import param_space
from train import train

tree = CommandTree()


def is_alphabetical_order(d, parent_keys=None):
    """Recursively check if all keys in the dictionary are in alphabetical order
    and return the keys that are out of order.
    """
    if parent_keys is None:
        parent_keys = []
    if not isinstance(d, dict):
        return []
    keys = list(d.keys())
    out_of_order_keys = [
        parent_keys + [k1] for k1, k2 in zip(keys, sorted(keys)) if k1 != k2
    ]
    for k, v in d.items():
        out_of_order_keys.extend(is_alphabetical_order(v, parent_keys + [k]))
    return out_of_order_keys


def check_alphabetical_order(d: DictConfig, name: str):
    out_of_order_keys = is_alphabetical_order(OmegaConf.to_container(d))
    if out_of_order_keys:
        print(f"The following keys are not in alphabetical order in {name}:")
        for keys in out_of_order_keys:
            print(".".join(keys))
        exit(1)


def get_config(config_name):
    root = Path("configs")
    config_path = root / f"{config_name}.yml"
    config = OmegaConf.load(config_path)
    check_alphabetical_order(config, str(config_path))
    base_config_path = config_path.parent / "base.yml"
    base_config = OmegaConf.load(base_config_path)
    check_alphabetical_order(base_config, str(base_config_path))
    merged = OmegaConf.merge(base_config, config)
    resolved = OmegaConf.to_container(merged, resolve=True)
    return resolved


def get_project_name():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["tool"]["poetry"]["name"]


def get_ignored() -> Set[Path]:
    repo = Repo(".")
    src = Path(__file__).parent
    ignored = repo.git.ls_files(
        str(src), o=True, i=True, exclude_standard=True
    ).splitlines()
    return set(Path(p).absolute() for p in ignored)


def check_dirty():
    assert not Repo(".").is_dirty()


def include_fn(path: str, exclude: list[str]):
    ignored = get_ignored()
    path = Path(path).absolute()
    include = path in ignored
    for pattern in exclude:
        include = include and not Path(path).match(pattern)
    return include


parsers = dict(config=option("config", default="delta-1"))


@tree.subcommand(parsers=dict(name=argument("name"), **parsers))
def log(
    config: str,
    name: str,
    allow_dirty: bool = False,
):
    if not allow_dirty:
        check_dirty()

    config_name = config
    config = get_config(config)
    run = wandb.init(
        config=dict(**config, config=config_name),
        name=name,
        project=get_project_name(),
    )

    train(**config, run=run)


@tree.command(parsers=parsers)
def no_log(
    config: str, load_path: str = None, memory_limit: int = None
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

    config = get_config(config)
    if load_path is not None:
        config = wandb.Api().run(load_path).config
    config.update(load_path=load_path)
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

    def trainable(sweep_params):
        sleep_time = 1
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
        config.update(run=run)
        return train(**config)

    tune.Tuner(
        trainable=tune.with_resources(trainable, dict(gpu=gpus_per_proc)),
        tune_config=None if num_samples is None else dict(num_samples=num_samples),
        param_space=param_space,
    ).fit()


if __name__ == "__main__":
    tree()
