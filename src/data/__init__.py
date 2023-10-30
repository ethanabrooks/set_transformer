import data.sample_trajectories
import data.sample_uniform
from data.dataset import MDP, Dataset
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
        mdp: MDP = data.sample_trajectories.MDP.make(**mdp_args)
        values = data.sample_trajectories.Values.make(
            mdp=mdp, stop_at_rmse=stop_at_rmse
        )
    elif sample_from == SampleFrom.UNIFORM:
        mdp: MDP = data.sample_uniform.MDP.make(**mdp_args)
        values: Dataset = data.sample_uniform.Values.make(
            mdp=mdp, stop_at_rmse=stop_at_rmse
        )
    dataset: Dataset = Dataset.make(**dataset_args, mdp=mdp, values=values)
    return dataset
