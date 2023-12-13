import torch

from ppo.envs.envs import EnvType
from ppo.rollout_storage import RolloutStorage
from train.utils import plot_trajectory


def plot(env_args: dict, env_type: EnvType, rollouts: RolloutStorage):
    if env_type == EnvType.SEQUENCE:
        obs: torch.Tensor = rollouts.obs[:-1, 0, : -env_args["n_objects"]].cpu()
        obs: torch.Tensor = obs.reshape(len(obs), -1, 3)
        l = obs.size(1)
        obs = obs[:, [[i] for i in range(l)], [[0, 2]]]
        *boxes, pos, dir_vec, _ = obs.unbind(1)
        masks = rollouts.masks[1:, 0].cpu().clone()
        masks[-1] = 0
        return plot_trajectory(
            boxes=boxes,
            done=masks == 0,
            pos=pos,
            dir_vec=dir_vec,
            q_vals=None,
            rewards=rollouts.rewards[:, 0].cpu(),
        )
