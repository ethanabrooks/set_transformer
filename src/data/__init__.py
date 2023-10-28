import data.sample_trajectories
import data.sample_uniform
from data.base import MDP, Dataset
from data.utils import MDPType


def make(
    dataset_args: dict,
    mdp_args: dict,
    name: str,
    seed: int,
) -> Dataset:
    mdp_type = MDPType[name.upper()]
    mdp_args.update(seed=seed)
    if mdp_type == MDPType.TRAJECTORIES:
        mdp: MDP = data.sample_trajectories.MDP.make(**mdp_args)
        dataset: Dataset = data.sample_trajectories.Dataset.make(
            **dataset_args, mdp=mdp
        )
    elif mdp_type == MDPType.UNIFORM:
        mdp: MDP = data.sample_uniform.MDP.make(**mdp_args)
        dataset: Dataset = data.sample_uniform.Dataset.make(**dataset_args, mdp=mdp)
    return dataset
