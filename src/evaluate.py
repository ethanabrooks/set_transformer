import functools
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gym.spaces import Box, Discrete, MultiDiscrete, Space
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tqdm import tqdm
from transformers import GPT2Config
from wandb.sdk.wandb_run import Run

import wandb
from envs.subproc_vec_env import SubprocVecEnv
from models.trajectories import GPT2, CausalTransformer
from sequence.base import Sequence
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
    epsilon: float,
    envs: SubprocVecEnv,
    iterations: int,
    net: CausalTransformer,
    rollout_length: int,
    x: DataPoint,
    ground_truth: torch.Tensor,
) -> pd.DataFrame:
    x_orig = x
    del x
    observation = envs.reset(state=x_orig.obs[:, 0].numpy())
    observation = torch.from_numpy(observation).float()
    assert torch.all(x_orig.obs[:, 0] == observation)
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
    optimals = None

    for i, o in enumerate(observation):
        optimal = envs.optimal(i, o)
        if optimal is not None:
            if optimals is None:
                optimals = torch.zeros_like(rewards)
            optimals[0, i] = optimal

    ground_truth = envs.values

    episode = torch.zeros(n, dtype=int)
    episodes = torch.zeros((l, n), dtype=int)
    timestep = torch.zeros(n, dtype=int)
    timesteps = torch.zeros((l, n), dtype=int)

    action_space = envs.action_space
    action_space.seed(0)

    input_q_zero = torch.zeros((context_length, n, a), dtype=float)
    idx_prefix = torch.arange(context_length - 1)

    for t in tqdm(range(l)):

        def check(*_):
            pass

        obs[t] = observation
        check(obs, x_orig.obs)

        if t < -1:  # context_length:
            # action = torch.tensor([action_space.sample() for _ in range(n)])
            # action_probs[t] = 1 / a
            # action_probs[t] = policy[torch.arange(n), observation.long()]
            # action = torch.multinomial(action_probs[t], 1).squeeze(-1)
            action_probs[t] = x_orig.action_probs[:, t]
            action = x_orig.actions[:, t]
        else:
            idx = torch.cat([idx_prefix, torch.tensor(t)[None]])
            input_q = input_q_zero
            x_T = DataPoint(*[y if y.ndim == 1 else y.swapaxes(0, 1) for y in x_orig])
            x_T.action_probs[-1] = fill_value
            x_T.actions[-1] = fill_value
            x_T.done[-1] = fill_value
            x_T.next_obs[-1] = fill_value
            x_T.rewards[-1] = fill_value
            x = DataPoint(
                action_probs=x_T.action_probs,  # action_probs[idx],
                actions=x_T.actions,  # actions[idx],
                done=x_T.done,  # dones[idx],
                idx=None,
                input_q=input_q,
                n_bellman=None,
                next_obs=x_T.next_obs,  # next_obs[idx],
                obs=x_T.obs,  # obs[idx],
                rewards=x_T.rewards,  # rewards[idx],
                target_q=None,
            )
            x = DataPoint(
                *[y if y is None else y[-context_length:].swapaxes(0, 1) for y in x]
            )

            input_q = torch.zeros_like(x.input_q)
            # errors = []
            # with torch.no_grad():
            #     for i in range(ground_truth.size(1) - 1):
            #         n_bellman = i * torch.ones(n).long()
            #         input_q: torch.Tensor
            #         input_q, _ = net.forward_with_rotation(
            #             x._replace(input_q=input_q, n_bellman=n_bellman),
            #             optimizer=None,
            #         )
            #         gt = ground_truth[torch.arange(n)[:, None], i + 1, x.obs.long()]
            #         # sar = torch.stack([x.obs, x.actions, x.rewards], -1)[0]
            #         # sep = float("nan") * torch.ones((context_length, 1))
            #         # gt = ground_truth_values[idx][:, :, 1][:, 0]
            #         # rounded = (input_q * 100).round().cpu()
            #         mae = (input_q.cpu() - gt).abs().mean()
            #         errors.append(mae.item())
            # output = input_q.cpu()
            # acc = (output.argmax(-1) == gt.argmax(-1)).float()
            # action = output[:, -1].argmax(-1)
            # print("\n".join(list(render_eval_metrics(*errors, max_num=1))))
            # print("Overall Accuracy:", acc.mean().item())
            # print("Last Index Accuracy:", acc[:, -1].mean().item())
            action = ground_truth[torch.arange(n), -1, observation.long()].argmax(-1)
            action_probs[t] = torch.eye(a)[action]
            # action = torch.multinomial(action_probs[t], 1).squeeze(-1)
        actions[t] = action
        info_list: list[dict]
        observation, reward, done, info_list = envs.step(action.numpy())
        step = StepResult(
            reward=reward, observation=observation, done=done, info=info_list
        )

        # record step result
        assert len(step.info) == n
        next_obs[t] = torch.from_numpy(step.observation)
        check(next_obs, x_orig.next_obs)
        rewards[t] = torch.from_numpy(step.reward)
        check(rewards, x_orig.rewards)
        dones[t] = torch.from_numpy(step.done)
        check(dones, x_orig.done)
        observation = torch.from_numpy(step.observation)

        # record episode timesteps
        timesteps[t] = timestep
        timestep += 1
        timestep[step.done] = 0
        episodes[t] = episode
        episode += step.done

        # check for done
        for index, done in enumerate(step.done):
            assert isinstance(done, (bool, np.bool_))
            if done:
                if t + 1 < x_orig.obs.shape[1]:
                    observation[index] = envs.reset(index, x_orig.obs[index, t + 1])
                    assert torch.all(observation[index] == x_orig.obs[index, t + 1])
                else:
                    observation[index] = envs.reset(index)

                optimal = envs.optimal(index, observation[index])
                if optimal is not None and t + 1 < len(optimals):
                    optimals[t + 1, index] = optimal
    idx = torch.arange(n)[None].expand(l, -1)
    data = dict(
        actions=actions,
        dones=dones,
        episode=episodes,
        idx=idx,
        obs=obs,
        rewards=rewards,
        timesteps=timesteps,
    )
    if optimals is not None:
        data["optimals"] = optimals

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
    count_threshold: int,
    df: pd.DataFrame,
    max_returns: float,
    run: Run,
    sequence: Sequence,
):
    def get_returns(key: str, df: pd.DataFrame):
        exponents = np.arange(len(df))
        discounts = np.power(sequence.grid_world.gamma, exponents)
        discounted = df[key] * discounts
        return discounted.sum()

    def is_complete(df: pd.DataFrame):
        dones: pd.Series = df["dones"]
        return dones.iloc[-1]

    complete_episodes = df.groupby(["episode", "idx"]).filter(is_complete)
    returns: pd.Series = complete_episodes.groupby(["episode", "idx"]).apply(
        functools.partial(get_returns, "rewards")
    )
    graphs = dict(returns=returns)
    if "optimals" in df.columns:
        optimals: pd.Series = complete_episodes.groupby(["episode", "idx"]).apply(
            functools.partial(get_returns, "optimals")
        )
        metrics = optimals - returns
        breakpoint()
        assert (metrics >= 0).all()
        graphs["regret"] = metrics
    plot_log = {}
    test_log = {}
    for name, metrics in graphs.items():
        means = metrics.groupby("episode").mean()
        sems = metrics.groupby("episode").sem()
        counts = metrics.groupby("episode").count()
        means = means[counts > count_threshold]
        sems = sems[counts > count_threshold]
        if run is None:
            graph = list(render_eval_metrics(*metrics, max_num=1))
            print(f"\n{name}\n" + "\n".join(graph), end="\n\n")
        breakpoint()

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()
        x = means.index + 1  # 1-indexed
        ax.fill_between(x, means - sems, means + sems, alpha=0.2)
        ax.plot(x, means)
        if name == "returns":
            ymin, ymax = None, max(means.max(), max_returns)
        elif name == "regret":
            ymin, ymax = 0, None
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("episode")
        ax.set_ylabel(name)
        ax.grid(True)

        test_log[name] = means.iloc[-1]
        plot_log[name] = wandb.Image(fig)
    return plot_log, test_log
