from sequence.grid_world_base import Sequence
from values.cross_product import Values as SampleCrossProductValues
from values.tabular import Values


def make(sample_from_trajectories: bool, sequence: Sequence, **kwargs) -> Values:
    kwargs.update(Q=None, sequence=sequence)

    return (
        Values.make(**kwargs)
        if sample_from_trajectories
        else SampleCrossProductValues.make(
            **kwargs, stop_at_rmse=sequence.grid_world.stop_at_rmse
        )
    )
