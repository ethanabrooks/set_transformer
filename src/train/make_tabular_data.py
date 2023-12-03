from dataset.tabular import Dataset
from grid_world.base import GridWorld
from sequence import make_grid_world_sequence
from sequence.grid_world_base import Sequence
from values import make as make_values


def make_sequence(
    grid_world_args: dict,
    seed: int,
    **kwargs: dict,
) -> Sequence:
    grid_world = GridWorld.make(**grid_world_args, seed=seed, terminal_transitions=None)
    kwargs.update(grid_world=grid_world)
    return make_grid_world_sequence(**kwargs)


def make_data(
    bellman_delta: int,
    dataset_args: dict,
    sequence_args: dict,
    sample_from_trajectories: bool,
    seed: int,
) -> Dataset:
    sequence: Sequence = make_sequence(
        partial_observation=False,  # Not implemented
        **sequence_args,
        sample_from_trajectories=sample_from_trajectories,
        seed=seed,
    )
    values = make_values(
        sequence=sequence, sample_from_trajectories=sample_from_trajectories
    )
    dataset: Dataset = Dataset(
        **dataset_args, bellman_delta=bellman_delta, sequence=sequence, values=values
    )
    return dataset
