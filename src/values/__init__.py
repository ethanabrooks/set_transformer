from sequence.base import Sequence
from values.base import Values
from values.sample_uniform import Values as SampleUniformValues

MODEL_FNAME = "model.tar"


def make(
    sequence: Sequence,
    sample_from_trajectories: bool,
) -> Values:
    return (
        Values.make(sequence=sequence)
        if sample_from_trajectories
        else SampleUniformValues.make(sequence=sequence)
    )
