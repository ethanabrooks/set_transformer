from sequence.base import Sequence
from values.sample_uniform import Values as SampleUniformValues
from values.tabular import Values

MODEL_FNAME = "model.tar"


def make(
    sequence: Sequence,
    sample_from_trajectories: bool,
    stop_at_rmse: float,
) -> Values:
    return (
        Values.make(sequence=sequence, stop_at_rmse=stop_at_rmse)
        if sample_from_trajectories
        else SampleUniformValues.make(sequence=sequence, stop_at_rmse=stop_at_rmse)
    )
