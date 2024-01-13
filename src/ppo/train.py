import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from gymnasium.spaces import Discrete
from torch.optim import Adam
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from ppo import utils
from ppo.agent import Agent
from ppo.data_storage import DataStorage
from ppo.envs.envs import EnvType, VecPyTorch, make_vec_envs
from ppo.rollout_storage import RolloutStorage
from utils import Transition, load, save


def infos_to_array(infos: list[dict], key: str) -> np.ndarray:
    def generate():
        for info in infos:
            state = info.get(key)
            if state is not None:
                yield state

    arrays = list(generate())
    if arrays:
        return np.stack(arrays)


def train(
    agent_args: dict,
    disable_gae: bool,
    disable_linear_lr_decay: bool,
    disable_proper_time_limits: bool,
    dummy_vec_env: bool,
    env_name: str,
    env_args: dict,
    gae_lambda: float,
    gamma: float,
    load_path: str,
    log_interval: int,
    lr: float,
    num_processes: int,
    num_steps: int,
    num_updates: int,
    optim_args: dict,
    run: Optional[Run],
    save_interval: int,
    seed: int,
    update_args: dict,
    use_replay_buffer: bool,
):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # noqa: Vulture
    torch.backends.cudnn.benchmark = False  # noqa: Vulture

    torch.set_num_threads(1)
    device = torch.device("cuda")
    env_type = EnvType[env_name]

    envs: VecPyTorch = make_vec_envs(
        device=device,
        dummy_vec_env=dummy_vec_env,
        **env_args,
        env_type=env_type,
        gamma=gamma,
        num_processes=num_processes,
        seed=seed,
    )

    use_tasks = envs.task_space is not None
    if use_tasks:
        agent_args.update(num_tasks=envs.task_space.n)
    else:
        agent_args.update(num_tasks=None)
    agent = Agent(
        obs_space=envs.observation_space,
        action_space=envs.action_space,
        **agent_args,
    )
    if load_path is not None:
        load(load_path, agent, run=None)
    agent.to(device)

    optimizer = Adam(agent.parameters(), lr=lr, **optim_args)
    rollouts = RolloutStorage(
        num_steps=num_steps,
        num_processes=num_processes,
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        recurrent_hidden_state_size=agent.recurrent_hidden_state_size,
    )

    obs, infos = envs.reset()
    state = infos_to_array(infos, "state")
    if state is None:
        state = obs
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = []
    df = pd.DataFrame(columns=["reward", "timestep"])

    def append_rewards(timestep: int, update: int):
        new_df = pd.DataFrame(
            dict(reward=[reward], timestep=[timestep], update=[update])
        )
        return pd.concat([df, new_df], ignore_index=True)

    start = time.time()

    action_space = envs.action_space
    if use_replay_buffer and isinstance(action_space, Discrete):
        replay_buffer_dir = Path("/tmp" if run is None else run.dir)
        action_probs_shape = (action_space.n,)
        data_storage = DataStorage.make(
            action_probs_shape=action_probs_shape,
            action_space=envs.action_space,
            num_timesteps=num_updates * num_steps,
            num_processes=num_processes,
            obs_space=envs.observation_space,
            path=DataStorage.make_path(replay_buffer_dir),
            state_shape=state.shape[1:],
        )
    else:
        data_storage = None

    for j in range(num_updates):
        if not disable_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                optimizer=optimizer,
                epoch=j,
                total_num_epochs=num_updates,
                initial_lr=lr,
            )

        for step in tqdm(range(num_steps), desc=f"Update {j}/{num_updates}"):
            # Sample actions
            with torch.no_grad():
                tasks = infos_to_array(infos, "task")
                action, action_metadata = agent.act(
                    inputs=rollouts.obs[step],
                    rnn_hxs=rollouts.recurrent_hidden_states[step],
                    masks=rollouts.masks[step],
                    **(
                        {}
                        if tasks is None
                        else dict(tasks=torch.from_numpy(tasks).to(device))
                    ),
                )

            prev_obs = obs

            # Obser reward and next obs
            obs, reward, done, truncated, infos = envs.step(action)
            next_state = infos_to_array(infos, "state")
            if next_state is None:
                next_state = obs

            terminal = done | truncated
            if data_storage is not None:
                action_probs = action_metadata.probs
                assert torch.allclose(action_probs.sum(dim=-1), torch.ones(1).cuda())
                transition = Transition(
                    states=state,
                    actions=action.squeeze(-1),
                    action_probs=action_probs,
                    next_states=next_state,
                    rewards=reward,
                    done=terminal,
                    obs=prev_obs,
                    next_obs=obs,
                )
                state = next_state

                def to_numpy():
                    for k, v in asdict(transition).items():
                        if isinstance(v, torch.Tensor):
                            v = v.cpu().numpy()
                        assert isinstance(v, np.ndarray)
                        yield k, v

                data_storage.insert(
                    timestep=j * num_steps + step,
                    transition=Transition[np.ndarray](**dict(to_numpy())),
                )

            total_num_steps = (j + 1) * num_processes * step
            info: dict
            for info in infos:
                if "episode" in info.keys():
                    reward = info["episode"]["r"]
                    episode_rewards.append(info["episode"]["r"])
                    df = append_rewards(timestep=total_num_steps, update=j)

            # If done then clean the history of observations.
            masks = torch.from_numpy(~(done | truncated))
            bad_masks = torch.from_numpy(~truncated)

            tasks = infos_to_array(infos, "task")
            if tasks is not None:
                tasks = torch.from_numpy(tasks).to(device)
            rollouts.insert(
                actions=action,
                bad_masks=bad_masks,
                log_probs=action_metadata.log_probs,
                masks=masks,
                obs=obs,
                rewards=reward,
                rnn_hxs=action_metadata.rnn_hxs,
                tasks=tasks,
                value=action_metadata.value,
            )

        with torch.no_grad():
            next_value = agent.get_value(
                inputs=rollouts.obs[-1],
                masks=rollouts.masks[-1],
                rnn_hxs=rollouts.recurrent_hidden_states[-1],
            ).detach()

        rollouts.compute_returns(
            next_value,
            not disable_gae,
            gamma,
            gae_lambda,
            not disable_proper_time_limits,
        )

        if load_path is None:
            value_loss, action_loss, dist_entropy = agent.update(
                optimizer=optimizer, rollouts=rollouts, **update_args
            )

        rollouts.after_update()

        log_metrics = j % log_interval == 0
        if load_path is None and log_metrics and len(episode_rewards) > 1:
            end = time.time()
            mean_reward = np.mean(episode_rewards)

            log = dict(
                updates=j,
                mean_reward=mean_reward,
                fps=int(total_num_steps / (end - start)),
                value_loss=value_loss,
                action_loss=action_loss,
                dist_entropy=dist_entropy,
            )
            episode_rewards = []
            if run is not None:
                run.log(log, step=total_num_steps)
        if (j + 1) % save_interval == 0 and run is not None:
            save(run, agent)

    return data_storage, df
