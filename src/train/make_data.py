from typing import Optional

from wandb.sdk.wandb_run import Run

from dataset.value_conditional import Dataset
from sequence import load_sequence
from sequence import make as make_sequence
from sequence.base import Sequence
from values import make as make_values
from values.neural import Values as NeuralValues


def make_data(
    bellman_delta: int,
    dataset_args: dict,
    neural_values: bool,
    q_load_path: Optional[str],
    run: Optional[Run],
    sequence_args: dict,
    sample_from_trajectories: bool,
    seed: int,
    value_args: dict,
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

    if neural_values:
        values = NeuralValues.make(run=run, sequence=sequence, **value_args)
    else:
        values = make_values(
            load_path=q_load_path,
            sequence=sequence,
            sample_from_trajectories=sample_from_trajectories,
        )
    dataset: Dataset = Dataset.make(
        bellman_delta=bellman_delta, **dataset_args, sequence=sequence, values=values
    )
    return dataset
