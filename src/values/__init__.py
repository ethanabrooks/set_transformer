from sequence.base import Sequence
from utils import SampleFrom
from values.sample_uniform import Values as SampleUniformValues
from values.tabular import Values

MODEL_FNAME = "model.tar"


def make(
    sequence: Sequence,
    name: str,
    stop_at_rmse: float,
) -> Values:
    sample_from = SampleFrom[name.upper()]
    if sample_from == SampleFrom.TRAJECTORIES:
        return Values.make(sequence=sequence, stop_at_rmse=stop_at_rmse)
    elif sample_from == SampleFrom.UNIFORM:
        return SampleUniformValues.make(sequence=sequence, stop_at_rmse=stop_at_rmse)
