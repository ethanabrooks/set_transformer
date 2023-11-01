from sequence.base import Sequence
from sequence.sample_trajectories import Sequence as SampleTrajectoriesSequence
from sequence.sample_uniform import Sequence as SampleUniformSequence


def make(sequence_args: dict, sample_from_trajectories: bool, seed: int) -> Sequence:
    sequence_args.update(seed=seed)
    return (
        SampleTrajectoriesSequence.make(**sequence_args)
        if sample_from_trajectories
        else SampleUniformSequence.make(**sequence_args)
    )
