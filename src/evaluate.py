import functools
from dataclasses import dataclass
from typing import Optional
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
from gym.spaces import Box, Discrete, MultiDiscrete, Space
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm
from transformers import GPT2Config
from wandb.sdk.wandb_run import Run

from envs.subproc_vec_env import SubprocVecEnv
from models.trajectories import GPT2, Model
from ppo.train import infos_to_array
from sequence.base import Sequence
from train.plot import plot_trajectories
from utils import DataPoint


def clamp(action: torch.Tensor, space: Space):
    if isinstance(space, Discrete):
        return torch.clamp(action, min=0, max=space.n - 1)
    elif isinstance(space, MultiDiscrete):
        return torch.clamp(action, min=0, max=space.nvec - 1)
    elif isinstance(space, Box):
        low = torch.tensor(space.low).cuda()
        high = torch.tensor(space.high).cuda()
        return torch.clamp(action, min=low, max=high)
    else:
        raise NotImplementedError


@dataclass(frozen=True)
class StepResult:
    reward: np.ndarray
    observation: np.ndarray
    done: np.ndarray
    info: list[dict]


def rollout(
    envs: SubprocVecEnv,
    gradual_randomness_decay: bool,
    iterations: int,
    net: Model,
    rollout_length: int,
) -> pd.DataFrame:
    observation = envs.reset()
    observation = torch.from_numpy(observation).float()
    n, *o = observation.shape
    a = envs.action_space.n
    sequence_network: GPT2 = net.sequence_network
    config: GPT2Config = sequence_network.gpt2.config
    context_length = config.n_positions
    l = context_length + rollout_length
    fill_value = net.pad_value

    action_probs = torch.full((l, n, a), fill_value, dtype=torch.float32)
    actions = torch.full((l, n), fill_value, dtype=torch.int64)
    dones = torch.full((l, n), fill_value, dtype=torch.bool)
    next_obs = torch.zeros((l, n, *o), dtype=torch.float32)
    obs = torch.full((l, n, *o), fill_value, dtype=torch.float32)
    rewards = torch.full((l, n), fill_value, dtype=torch.float32)
    Q = torch.full((l, n, a), fill_value, dtype=torch.float32)
    optimals = None
    states = None

    for i, o in enumerate(observation):
        optimal = envs.optimal(i, o)
        if optimal is not None:
            if optimals is None:
                optimals = torch.zeros_like(rewards)
            optimals[0, i] = optimal

    episode = torch.zeros(n, dtype=int)
    episodes = torch.zeros((l, n), dtype=int)
    timestep = torch.zeros(n, dtype=int)
    timesteps = torch.zeros((l, n), dtype=int)

    action_space = envs.action_space
    action_space.seed(0)

    input_q_zero = torch.zeros((context_length, n, a), dtype=float)

    for t in tqdm(range(l)):
        obs[t] = observation

        if t < context_length:
            action_probs[t] = 1 / a
            action = torch.multinomial(action_probs[t], 1).squeeze(-1)
        else:
            idx = slice(t + 1 - context_length, t + 1)
            input_q = input_q_zero
            x = DataPoint(
                action_probs=action_probs[idx],
                actions=actions[idx],
                done=dones[idx],
                idx=None,
                input_q=input_q,
                n_bellman=None,
                next_obs=next_obs[idx],
                obs=obs[idx],
                rewards=rewards[idx],
                target_q=None,
            )
            x = DataPoint(*[y if y is None else y.swapaxes(0, 1) for y in x])

            input_q = torch.zeros_like(x.input_q)
            with torch.no_grad():
                for i in range(iterations):
                    n_bellman = i * torch.ones(n).long()
                    input_q: torch.Tensor
                    input_q, _ = net.forward_with_rotation(
                        x._replace(input_q=input_q, n_bellman=n_bellman), optimizer=None
                    )
            output = input_q.cpu()
            Q[t] = output[:, -1]
            best_action = Q[t].argmax(-1)
            action_prob = torch.eye(a)[best_action]
            if gradual_randomness_decay:
                randomness = ((l - t) / rollout_length) ** 2
                action_probs[t] = (1 - randomness) * action_prob + randomness / a
            else:
                action_probs[t] = action_prob
            action = torch.multinomial(action_probs[t], 1).squeeze(-1)

        actions[t] = action
        info_list: list[dict]
        observation, reward, done, info_list = envs.step(action.numpy())
        step = StepResult(
            reward=reward, observation=observation, done=done, info=info_list
        )

        # record step result
        assert len(step.info) == n
        next_obs[t] = torch.from_numpy(step.observation)
        rewards[t] = torch.from_numpy(step.reward)
        dones[t] = torch.from_numpy(step.done)
        observation = torch.from_numpy(step.observation)
        state = infos_to_array(info_list, "state")
        if state is not None:
            if states is None:
                _, *s = state.shape
                states = torch.full((l, n, *s), fill_value, dtype=torch.float32)
            states[t] = torch.from_numpy(state)

        # record episode timesteps
        timesteps[t] = timestep
        timestep += 1
        episodes[t] = episode
        episode += step.done

        # check for done
        for index, done in enumerate(step.done):
            assert isinstance(done, (bool, np.bool_))
            if done:
                reset_obs = envs.reset(index)
                observation[index] = torch.from_numpy(np.array(reset_obs))

                optimal = envs.optimal(index, reset_obs)
                if optimal is not None and t + 1 < len(optimals):
                    optimals[t + 1, index] = optimal
    idx = torch.arange(n)[None].expand(l, -1)
    data = dict(
        actions=actions,
        dones=dones,
        episode=episodes,
        idx=idx,
        obs=obs,
        Q=Q[
            torch.arange(l)[:, None],
            torch.arange(n)[None, :],
            actions,
        ],
        rewards=rewards,
        timesteps=timesteps,
    )
    for k, v in dict(optimals=optimals, states=states).items():
        if v is not None:
            data[k] = v

    def process(x: torch.Tensor) -> "int | float | np.ndarray":
        x = x.reshape(n * l, -1).squeeze(-1)
        if x.ndim == 1:
            return x.numpy()
        elif x.ndim == 2:
            return list(x.numpy())
        else:
            raise ValueError

    return pd.DataFrame({k: process(v) for k, v in data.items()})


def render_eval_metrics(
    *numbers: float, max_num: Optional[float] = None, width: int = 10, length: int = 10
):
    if len(numbers) > length:
        subarrays = np.array_split(numbers, length)
        numbers = [subarray.mean() for subarray in subarrays]

    bar_elements = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
    min_num = min(numbers)
    if max_num is None:
        max_num = max(numbers)
    data_range = max(numbers) - min_num

    if data_range == 0:
        precision = 1
    else:
        smallest_unit = data_range / (width * (len(bar_elements) - 1))
        precision = max(0, -int(np.floor(np.log10(smallest_unit))))

    for num in numbers:
        num = min(num, max_num)
        ratio = (num - min_num) / (max_num - min_num) if max_num != min_num else 1
        full_blocks = int(ratio * width)
        fraction = ratio * width - full_blocks
        bar = full_blocks * "█"
        partial_block = round(fraction * (len(bar_elements) - 1))
        if num < max_num:
            bar += bar_elements[partial_block]
        padding = width - len(bar)
        padding = " " * padding
        num = round(num, precision)
        yield f"{num:<6} {bar}{padding}▏"


def log(
    df: pd.DataFrame,
    run: Run,
    sequence: Sequence,
    step: int,
):
    def get_returns(
        key: str,
        gamma: float,
        df: pd.DataFrame,
    ):
        exponents = np.arange(len(df))
        discounts = np.power(gamma, exponents)
        discounted = df[key] * discounts
        return discounted.sum()

    def is_complete(df: pd.DataFrame):
        dones: pd.Series = df["dones"]
        return dones.iloc[-1]

    complete_episodes = df.groupby(["episode", "idx"]).filter(is_complete)
    discounted_returns: pd.Series = complete_episodes.groupby(["episode", "idx"]).apply(
        functools.partial(get_returns, "rewards", sequence.gamma)
    )
    undiscounted_returns = complete_episodes.groupby(["episode", "idx"]).apply(
        functools.partial(get_returns, "rewards", 1)
    )
    last_timesteps = complete_episodes.groupby(["episode", "idx"])["timesteps"].last()
    episode_df = pd.concat(
        {
            "discounted returns": discounted_returns,
            "returns": undiscounted_returns,
            "timesteps": last_timesteps,
        },
        axis=1,
    )
    episode_df["step"] = step
    names = ["returns", "discounted returns"]
    if "optimals" in df.columns:
        optimals: pd.Series = complete_episodes.groupby(["episode", "idx"]).apply(
            functools.partial(get_returns, "optimals", sequence.gamma)
        )
        metrics = optimals - discounted_returns
        # assert (metrics >= 0).all()
        episode_df["regret"] = metrics
        names.append("regret")
    plot_log = {}
    test_log = {}
    if "states" in df.columns:
        first_index = complete_episodes[
            complete_episodes.idx == complete_episodes.idx.iloc[0]
        ]

        def tensor(k: str):
            return torch.from_numpy(first_index[k].to_numpy())[None]

        for i, fig in enumerate(
            plot_trajectories(
                done=tensor("dones"),
                Q=tensor("Q"),
                rewards=tensor("rewards"),
                states=torch.from_numpy(np.stack(first_index["states"]))[None],
            )
        ):
            # fig.savefig(f"plot_{i}.png")
            plot_log[f"trajectories/{i}"] = wandb.Image(fig)

    for name in names:
        metrics = episode_df[name]
        means = metrics.groupby("episode").mean()
        sems = metrics.groupby("episode").sem()
        if run is None:
            graph = list(render_eval_metrics(*metrics, max_num=1))
            print(f"\n{name}\n" + "\n".join(graph), end="\n\n")

        if means.empty:
            warn(f"No complete episodes for {name}.")
            return plot_log, test_log

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()
        x = means.index + 1  # 1-indexed
        ax.fill_between(x, means - sems, means + sems, alpha=0.2)
        ax.plot(x, means)
        if name == "returns":
            ymin, ymax = None, means.max()
        elif name == "regret":
            ymin, ymax = 0, None
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("episode")
        ax.set_ylabel(name)
        ax.grid(True)

        test_log[name] = means.iloc[-1]
        plot_log[name] = wandb.Image(fig)

    plot_log["table"] = wandb.Table(dataframe=episode_df)
    return plot_log, test_log
