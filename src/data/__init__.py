import data.sample_trajectories
import data.sample_uniform
from data.dataset import MDP, Dataset
from data.utils import SampleFrom


def make(
    dataset_args: dict,
    mdp_args: dict,
    name: str,
    seed: int,
) -> Dataset:
    sample_from = SampleFrom[name.upper()]
    mdp_args.update(seed=seed)
    if sample_from == SampleFrom.TRAJECTORIES:
        mdp: MDP = data.sample_trajectories.MDP.make(**mdp_args)
        dataset: Dataset = data.sample_trajectories.Dataset.make(
            **dataset_args, mdp=mdp
        )
    elif sample_from == SampleFrom.UNIFORM:
        mdp: MDP = data.sample_uniform.MDP.make(**mdp_args)
        dataset: Dataset = data.sample_uniform.Dataset.make(**dataset_args, mdp=mdp)
    return dataset
