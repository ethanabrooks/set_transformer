import time
from collections import deque
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
from torch.optim import Adam
from wandb.sdk.wandb_run import Run

from ppo import utils
from ppo.agent import Agent
from ppo.envs.envs import make_vec_envs
from ppo.storage import RolloutStorage
from ppo.utils import get_vec_normalize


def train(
    disable_gae: bool,
    disable_linear_lr_decay: bool,
    disable_proper_time_limits: bool,
    dummy_vec_env: bool,
    env_name: str,
    gae_lambda: float,
    gamma: float,
    load_path: str,
    log_interval: int,
    lr: float,
    num_processes: int,
    num_steps: int,
    num_env_steps: int,
    optim_args: dict,
    recurrent_policy: bool,
    run: Optional[Run],
    seed: int,
    update_args: dict,
):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # noqa: Vulture
    torch.backends.cudnn.benchmark = False  # noqa: Vulture

    torch.set_num_threads(1)
    device = torch.device("cuda")

    envs = make_vec_envs(
        device=device,
        dummy_vec_env=dummy_vec_env,
        env_name=env_name,
        gamma=gamma,
        log_dir=None,
        num_processes=num_processes,
        seed=seed,
    )

    agent = Agent(
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        base_kwargs={"recurrent": recurrent_policy},
    )
    if load_path is not None:
        state_dict: dict = torch.load(load_path)
        vec_norm = get_vec_normalize(envs)
        if vec_norm is not None:
            ob_rms = state_dict.pop("ob_rms")
            vec_norm.eval()
            vec_norm.ob_rms = ob_rms
        agent.load_state_dict(state_dict)
    agent.to(device)

    optimizer = Adam(agent.parameters(), lr=lr, **optim_args)
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        recurrent_hidden_state_size=agent.recurrent_hidden_state_size,
    )

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(num_env_steps) // num_steps // num_processes
    for j in range(num_updates):
        if not disable_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                optimizer=optimizer,
                epoch=j,
                total_num_epochs=num_updates,
                initial_lr=lr,
            )

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                action, action_metadata = agent.act(
                    inputs=rollouts.obs[step],
                    rnn_hxs=rollouts.recurrent_hidden_states[step],
                    masks=rollouts.masks[step],
                )

            # Obser reward and next obs
            obs, reward, done, truncated, infos = envs.step(action)

            info: dict
            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.from_numpy(~(done | truncated))
            bad_masks = torch.from_numpy(~truncated)

            rollouts.insert(
                obs=obs,
                actions=action,
                rewards=reward,
                masks=masks,
                bad_masks=bad_masks,
                **asdict(action_metadata)
            )

        with torch.no_grad():
            next_value = agent.get_value(
                rollouts.obs[-1],
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value,
            not disable_gae,
            gamma,
            gae_lambda,
            not disable_proper_time_limits,
        )

        value_loss, action_loss, dist_entropy = agent.update(
            optimizer=optimizer, rollouts=rollouts, **update_args
        )

        rollouts.after_update()

        if j % log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * num_processes * num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".format(
                    j,
                    total_num_steps,
                    int(total_num_steps / (end - start)),
                    len(episode_rewards),
                    np.mean(episode_rewards),
                    np.median(episode_rewards),
                    np.min(episode_rewards),
                    np.max(episode_rewards),
                )
            )
            log = dict(
                updates=j,
                mean_reward=np.mean(episode_rewards),
                fps=int(total_num_steps / (end - start)),
                value_loss=value_loss,
                action_loss=action_loss,
                dist_entropy=dist_entropy,
            )
            if run is not None:
                run.log(log, step=total_num_steps)
