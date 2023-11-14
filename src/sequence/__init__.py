import pickle
from pathlib import Path

from artifacts import download_and_return_root
from sequence.base import Sequence
from sequence.sample_trajectories import Sequence as SampleTrajectoriesSequence
from sequence.sample_uniform import Sequence as SampleUniformSequence


def load_sequence(
    load_path: str, **_
) -> Sequence:  # TODO: think of a better way to handle excess args
    type = "sequence"
    artifact_root = download_and_return_root(load_path=load_path, type=type)
    artifact_path: Path = artifact_root / f"{type}.pkl"
    with artifact_path.open("rb") as f:
        return pickle.load(f)


def make_sequence(sample_from_trajectories: bool, **kwargs: dict) -> Sequence:
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
