import time
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from gymnasium.spaces import Discrete
from torch.optim import Adam
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from ppo import utils
from ppo.agent import Agent
from ppo.data_storage import DataStorage
from ppo.envs.envs import VecPyTorch, make_vec_envs
from ppo.rollout_storage import RolloutStorage
from ppo.utils import get_vec_normalize
from utils import Transition


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
    seed: int,
    update_args: dict,
):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # noqa: Vulture
    torch.backends.cudnn.benchmark = False  # noqa: Vulture

    torch.set_num_threads(1)
    device = torch.device("cuda")

    envs: VecPyTorch = make_vec_envs(
        device=device,
        dummy_vec_env=dummy_vec_env,
        **env_args,
        gamma=gamma,
        num_processes=num_processes,
        seed=seed,
    )

    agent = Agent(
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        **agent_args,
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

    obs, infos = envs.reset()
    state = infos_to_array(infos, "state")
    if state is None:
        state = obs
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()

    action_space = envs.action_space
    if isinstance(action_space, Discrete):
        replay_buffer_dir = Path("/tmp" if run is None else run.dir)
        action_probs_shape = (action_space.n,)
        data_storage = DataStorage.make(
            action_dtype=envs.action_space.dtype,
            action_probs_shape=action_probs_shape,
            num_timesteps=num_updates * num_steps,
            num_processes=num_processes,
            obs_shape=envs.observation_space.shape,
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
                action, action_metadata = agent.act(
                    inputs=rollouts.obs[step],
                    rnn_hxs=rollouts.recurrent_hidden_states[step],
                    masks=rollouts.masks[step],
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

            info: dict
            for info in infos:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # If done then clean the history of observations.
            masks = torch.from_numpy(~(done | truncated))
            bad_masks = torch.from_numpy(~truncated)

            rollouts.insert(
                actions=action,
                bad_masks=bad_masks,
                log_probs=action_metadata.log_probs,
                masks=masks,
                obs=obs,
                rewards=reward,
                rnn_hxs=action_metadata.rnn_hxs,
                tasks=torch.from_numpy(infos_to_array(infos, "task")).to(device),
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

        value_loss, action_loss, dist_entropy = agent.update(
            optimizer=optimizer, rollouts=rollouts, **update_args
        )

        rollouts.after_update()

        if j % log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * num_processes * num_steps
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
            if run is not None:
                run.log(log, step=total_num_steps)

    return data_storage
