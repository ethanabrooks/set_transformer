from typing import Optional

from dataset.tabular import Dataset
from sequence import load_sequence
from sequence import make as make_sequence
from sequence.base import Sequence
from values import make as make_values


def make_data(
    bellman_delta: int,
    dataset_args: dict,
    q_load_path: Optional[str],
    sequence_args: dict,
    sample_from_trajectories: bool,
    seed: int,
) -> Dataset:
    try:
        sequence: Sequence = load_sequence(**sequence_args)
    except TypeError:
        sequence: Sequence = make_sequence(
            partial_observation=False,  # Not implemented
            **sequence_args,
            sample_from_trajectories=sample_from_trajectories,
            seed=seed
        )
    values = make_values(
        load_path=q_load_path,
        sequence=sequence,
        sample_from_trajectories=sample_from_trajectories,
    )
    dataset: Dataset = Dataset(
        **dataset_args, bellman_delta=bellman_delta, sequence=sequence, values=values
    )
    return dataset
