from typing import Optional

from dataset.tabular import Dataset
from sequence import make as make_sequence
from sequence.grid_world_base import Sequence
from values import make as make_values


def make_data(
    bellman_delta: int,
    dataset_args: dict,
    q_load_path: Optional[str],
    sequence_args: dict,
    sample_from_trajectories: bool,
    seed: int,
) -> Dataset:
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
        **dataset_args,
        bellman_delta=bellman_delta,
        n_actions=len(sequence.grid_world.deltas),
        sequence=sequence,
        values=values
    )
    return dataset
