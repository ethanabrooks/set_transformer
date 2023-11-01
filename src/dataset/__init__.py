from dataset.base import Dataset
from dataset.sample_trajectories import Sequence as SampleTrajectoriesMDP
from dataset.sample_trajectories import Values as SampleTrajectoriesValues
from dataset.sample_uniform import Sequence as SampleUniformMDP
from dataset.sample_uniform import Values as SampleUniformValues
from sequence.base import Sequence
from utils import SampleFrom


def make(
    dataset_args: dict,
    sequence_args: dict,
    name: str,
    seed: int,
    stop_at_rmse: float,
) -> Dataset:
    sample_from = SampleFrom[name.upper()]
    sequence_args.update(seed=seed)
    if sample_from == SampleFrom.TRAJECTORIES:
        sequence: Sequence = SampleTrajectoriesMDP.make(**sequence_args)
        values = SampleTrajectoriesValues.make(
            sequence=sequence, stop_at_rmse=stop_at_rmse
        )
    elif sample_from == SampleFrom.UNIFORM:
        sequence: Sequence = SampleUniformMDP.make(**sequence_args)
        values: Dataset = SampleUniformValues.make(
            sequence=sequence, stop_at_rmse=stop_at_rmse
        )
    dataset: Dataset = Dataset.make(**dataset_args, sequence=sequence, values=values)
    return dataset
