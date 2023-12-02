from grid_world.base import GridWorld
from sequence.base import Sequence
from sequence.cross_product import Sequence as SampleUniformSequence
from sequence.trajectories import Sequence as SampleTrajectoriesSequence


def make_sequence(
    sample_from_trajectories: bool,
    **kwargs: dict,
) -> Sequence:
    return (
        SampleTrajectoriesSequence.make(**kwargs)
        if sample_from_trajectories
        else SampleUniformSequence.make(**kwargs)
    )


def make(
    grid_world_args: dict,
    seed: int,
    **kwargs: dict,
) -> Sequence:
    grid_world = GridWorld.make(**grid_world_args, seed=seed, terminal_transitions=None)
    kwargs.update(grid_world=grid_world)
    return make_sequence(**kwargs)
