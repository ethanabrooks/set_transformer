from sequence.sample_trajectories import Sequence as SampleTrajectoriesSequence
from sequence.sample_uniform import Sequence as SampleUniformSequence
from sequence.base import Sequence
from utils import SampleFrom


def make(sequence_args: dict, name: str, seed: int) -> Sequence:
    sample_from = SampleFrom[name.upper()]
    sequence_args.update(seed=seed)
    if sample_from == SampleFrom.TRAJECTORIES:
        return SampleTrajectoriesSequence.make(**sequence_args)
    elif sample_from == SampleFrom.UNIFORM:
        return SampleUniformSequence.make(**sequence_args)
