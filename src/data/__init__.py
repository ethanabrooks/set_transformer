from data.dataset import Dataset
from data.mdp import MDP
from data.sample_trajectories import MDP as SampleTrajectoriesMDP
from data.sample_trajectories import Values as SampleTrajectoriesValues
from data.sample_uniform import MDP as SampleUniformMDP
from data.sample_uniform import Values as SampleUniformValues
from data.utils import SampleFrom


def make(
    dataset_args: dict,
    mdp_args: dict,
    name: str,
    seed: int,
    stop_at_rmse: float,
) -> Dataset:
    sample_from = SampleFrom[name.upper()]
    mdp_args.update(seed=seed)
    if sample_from == SampleFrom.TRAJECTORIES:
        mdp: MDP = SampleTrajectoriesMDP.make(**mdp_args)
        values = SampleTrajectoriesValues.make(mdp=mdp, stop_at_rmse=stop_at_rmse)
    elif sample_from == SampleFrom.UNIFORM:
        mdp: MDP = SampleUniformMDP.make(**mdp_args)
        values: Dataset = SampleUniformValues.make(mdp=mdp, stop_at_rmse=stop_at_rmse)
    dataset: Dataset = Dataset.make(**dataset_args, mdp=mdp, values=values)
    return dataset
