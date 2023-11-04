from sequence.base import Sequence
from values.sample_uniform import Values as SampleUniformValues
from values.tabular import Values

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
