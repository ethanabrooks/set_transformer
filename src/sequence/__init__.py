from sequence.base import Sequence
from sequence.sample_trajectories import Sequence as SampleTrajectoriesSequence
from sequence.sample_uniform import Sequence as SampleUniformSequence


def make(sample_from_trajectories: bool, **kwargs: dict) -> Sequence:
    return (
        SampleTrajectoriesSequence.make(**kwargs)
        if sample_from_trajectories
        else SampleUniformSequence.make(**kwargs)
    )
