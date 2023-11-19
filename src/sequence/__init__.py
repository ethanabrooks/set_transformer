import pickle
from pathlib import Path

from artifacts import download_and_return_root
from sequence.base import Sequence
from sequence.sample_trajectories import Sequence as SampleTrajectoriesSequence
from sequence.sample_uniform import Sequence as SampleUniformSequence
from tabular.grid_world import GridWorld


def load_sequence(
    load_path: str, **_
) -> Sequence:  # TODO: think of a better way to handle excess args
    type = "sequence"
    artifact_root = download_and_return_root(load_path=load_path, type=type)
    artifact_path: Path = artifact_root / f"{type}.pkl"
    with artifact_path.open("rb") as f:
        return pickle.load(f)


def make_sequence(
    grid_world_args: dict,
    sample_from_trajectories: bool,
    seed: int,
    **kwargs: dict,
) -> Sequence:
    grid_world = GridWorld.make(**grid_world_args, seed=seed, terminal_transitions=None)
    kwargs.update(grid_world=grid_world)
    return (
        SampleTrajectoriesSequence.make(**kwargs)
        if sample_from_trajectories
        else SampleUniformSequence.make(**kwargs)
    )


def make(**kwargs: dict) -> Sequence:
    try:
        return load_sequence(**kwargs)
    except TypeError:
        return make_sequence(**kwargs)
