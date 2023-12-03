from sequence.cross_product import Sequence as SampleUniformSequence
from sequence.grid_world_base import Sequence
from sequence.trajectories import Sequence as SampleTrajectoriesSequence


def make_grid_world_sequence(
    sample_from_trajectories: bool,
    **kwargs: dict,
) -> Sequence:
    return (
        SampleTrajectoriesSequence.make(**kwargs)
        if sample_from_trajectories
        else SampleUniformSequence.make(**kwargs)
    )
