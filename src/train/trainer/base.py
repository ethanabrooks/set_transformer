import itertools
import math
import time
from dataclasses import asdict, dataclass, replace
from typing import Counter, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

import wandb
from dataset.trajectories import Dataset
from envs.subproc_vec_env import SubprocVecEnv
from evaluate import log as log_evaluation
from evaluate import rollout
from metrics import Metrics, compute_rmse, get_metrics
from models.trajectories import Model
from sequence.grid_world_base import Sequence
from utils import DataPoint, decay_lr, load, save
from values.bootstrap import Values as BootstrapValues


@dataclass
class Evaluator:
    envs: SubprocVecEnv
    net: Model
    rollout_length: int

    def rollout(self, iterations: int) -> pd.DataFrame:
        return rollout(
            envs=self.envs,
            iterations=iterations,
            net=self.net,
            rollout_length=self.rollout_length,
        )


def train_bellman_iteration(
    alpha: float,
    baseline: bool,
    bellman_delta: int,
    bellman_number: int,
    bootstrap_Q: torch.Tensor,
    count_threshold: int,
    decay_args: dict,
    evaluator: Evaluator,
    load_path: str,
    log_interval: int,
    lr: float,
    metrics_args: dict,
    n_batch: int,
    net: Model,
    optimizer: optim.Optimizer,
    plot_indices: torch.Tensor,
    run: Run,
    sequence: Sequence,
    start_step: int,
    stop_at_rmse: float,
    test_interval: int,
):
    if run is not None:
        wandb.log(dict(bellman_number=bellman_number), step=start_step)

    assert torch.all(bootstrap_Q[0] == 0)
    Q = sequence.grid_world.Q
    ground_truth = Q[1 : 1 + bellman_number]  # omit Q_0 from ground_truth

    test_log = {}
    counter = Counter()
    Q = bootstrap_Q.clone()
    _, _, l, a = bootstrap_Q.shape
    updated = None

    def make_dataset(bootstrap_Q: torch.Tensor):
        assert len(bootstrap_Q) == bellman_number
        values = BootstrapValues.make(bootstrap_Q=bootstrap_Q, sequence=sequence)
        return Dataset(
            bellman_delta=bellman_delta, n_actions=a, sequence=sequence, values=values
        )

    def _get_metrics(prefix: str, outputs: torch.Tensor, targets: torch.Tensor):
        metrics = get_metrics(
            loss=None, outputs=outputs, targets=targets, **metrics_args
        )
        return {f"{prefix}/{k}": v for k, v in asdict(metrics).items()}

    def update_plots():
        grid_world = sequence.grid_world
        q_per_state = torch.empty(len(plot_indices), grid_world.n_states, a)
        q_per_state[
            torch.arange(len(plot_indices))[:, None],
            sequence.transitions.states[plot_indices],
        ] = Q[-1, plot_indices]
        stacked = torch.stack([q_per_state, grid_world.Q[bellman_number, plot_indices]])

        Pi = grid_world.Pi[None, plot_indices]
        v_per_state: torch.Tensor = stacked * Pi
        v_per_state = v_per_state.sum(-1)
        v_per_state = torch.unbind(v_per_state, dim=1)
        for i, plot_value in enumerate(v_per_state):
            fig = grid_world.visualize_values(plot_value)
            if wandb.run is not None:
                run.log({f"plot {i}/bellman {bellman_number}": wandb.Image(fig)})

    train_data = make_dataset(bootstrap_Q)
    tick = time.time()

    for e in itertools.count():
        if updated is not None:
            assert torch.all(updated)
        updated = torch.zeros_like(Q)

        q, b, l, a = Q.shape
        epoch_rmse = compute_rmse(
            train_data.values.Q,
            Q[
                torch.arange(q)[:, None, None],
                torch.arange(b)[None, :, None],
                torch.arange(l)[None, None],
                sequence.transitions.actions[None],
            ],
        )

        idxs = (
            torch.arange(q)[:, None, None],
            torch.arange(b)[None, :, None],
            sequence.transitions.states[None],
        )
        if q <= len(ground_truth):
            ground_truth_metrics = _get_metrics(
                "(ground-truth)", outputs=Q, targets=ground_truth[idxs]
            )

            versus_metrics = _get_metrics(
                "(bootstrap versus ground-truth)",
                outputs=train_data.values.Q,
                targets=ground_truth[[*idxs, sequence.transitions.actions[None]]],
            )
        else:
            ground_truth_metrics = {}
            versus_metrics = {}

        if baseline:
            bootstrap_Q2 = Q
        else:
            bootstrap_Q2 = F.pad(Q, (0, 0, 0, 0, 0, 0, 1, 0))[:-1]
        bootstrap_Q = alpha * bootstrap_Q2 + (1 - alpha) * bootstrap_Q
        train_data = make_dataset(bootstrap_Q)
        train_loader = DataLoader(train_data, batch_size=n_batch, shuffle=True)
        epoch_step = start_step + e * len(train_loader)
        done = epoch_rmse <= stop_at_rmse
        if done:
            update_plots()
            save(run, net)
        if baseline:
            run_evaluation = e % test_interval == 0
        else:
            run_evaluation = done and (bellman_number % test_interval == 0)
        if bellman_number == 1 and e == 0:
            run_evaluation = True
        if evaluator is None:
            run_evaluation = False
        if run_evaluation:
            df = evaluator.rollout(iterations=math.ceil(bellman_number / bellman_delta))
            plot_log, test_log = log_evaluation(
                count_threshold=count_threshold,
                df=df,
                run=run,
                sequence=sequence,
            )
            if run is not None:
                log = dict(**plot_log, **{f"test/{k}": v for k, v in test_log.items()})
                wandb.log(log, step=epoch_step)
        x: DataPoint
        net.train()
        for t, x in enumerate(train_loader):
            step = epoch_step + t
            decayed_lr = decay_lr(lr, step=step, **decay_args)
            if done or (step % log_interval == 0):
                fps = log_interval / (time.time() - tick)
                tick = time.time()
                train_log = {k: v / counter["n"] for k, v in counter.items()}
                test_log.update(bellman_number=bellman_number)
                repeated_log = {f"repeat/{k}": v for k, v in test_log.items()}
                update_plots()

                train_log.update(
                    epoch=e,
                    epoch_rmse=epoch_rmse,
                    fps=fps,
                    **ground_truth_metrics,
                    lr=decayed_lr,
                    max_Q=train_data.values.Q.max().item(),
                    **versus_metrics,
                )
                train_log = {f"train/{k}": v for k, v in train_log.items()}
                counter = Counter()
                print(".", end="", flush=True)
                if run is not None:
                    wandb.log(dict(**repeated_log, **train_log), step=step)
            if done:
                return Q, epoch_step

            for param_group in optimizer.param_groups:
                param_group.update(lr=decayed_lr)
            outputs, loss = net.forward_with_rotation(
                x=x, optimizer=optimizer if load_path is None else None
            )
            idxs = x.n_bellman, x.idx
            outputs_cpu = outputs.detach().cpu()
            Q[idxs] = outputs_cpu
            updated[idxs] = 1
            b, l, _ = outputs.shape
            metrics: Metrics = get_metrics(
                loss=loss,
                outputs=outputs_cpu[
                    torch.arange(b)[:, None],
                    torch.arange(l)[None],
                    x.actions,
                ],
                targets=x.target_q,
                **metrics_args,
            )
            counter.update(asdict(metrics), n=1)


def compute_values(
    baseline: bool,
    bellman_delta: int,
    envs: SubprocVecEnv,
    evaluator_args: dict,
    lr: float,
    model_args: dict,
    n_plot: int,
    partial_observation: bool,
    rmse_bellman: float,
    rmse_training_final: float,
    rmse_training_intermediate: float,
    run: Run,
    sequence: Sequence,
    train_args: dict,
    load_path: Optional[str] = None,
):
    grid_world = sequence.grid_world
    if baseline:
        grid_world = replace(grid_world, Q=grid_world.Q[[0, -1]])
        sequence = replace(sequence, grid_world=grid_world)
    b, l = sequence.transitions.rewards.shape
    a = envs.action_space.n
    Q = torch.zeros(1, b, l, a)
    values = BootstrapValues.make(sequence=sequence, bootstrap_Q=Q)
    data = Dataset(
        bellman_delta=bellman_delta, n_actions=a, sequence=sequence, values=values
    )
    start_step = 0
    n_tokens = max(data.n_tokens, len(sequence.grid_world.Q) * 2)  # double for padding
    net = Model(
        bellman_delta=bellman_delta,
        **model_args,
        n_actions=data.n_actions,
        n_ctx=l,
        n_tokens=n_tokens,
        partial_observation=partial_observation,
        pad_value=data.pad_value,
    )
    if load_path is not None:
        load(load_path, net, run)
    net = net.cuda()

    evaluator = (
        Evaluator(envs=envs, net=net, **evaluator_args)
        if isinstance(net, Model)
        else None
    )
    optimizer = optim.Adam(net.parameters(), lr=lr)
    plot_indices = torch.randint(0, b, (n_plot,))
    final = baseline

    for bellman_number in itertools.count(1):
        new_Q, step = train_bellman_iteration(
            baseline=baseline,
            bellman_delta=bellman_delta,
            bellman_number=bellman_number,
            bootstrap_Q=Q,
            evaluator=evaluator,
            load_path=load_path,
            lr=lr,
            net=net,
            optimizer=optimizer,
            plot_indices=plot_indices,
            run=run,
            sequence=sequence,
            start_step=start_step,
            stop_at_rmse=rmse_training_final if final else rmse_training_intermediate,
            **train_args,
        )
        start_step = step
        rmse = compute_rmse(Q[-1], new_Q[-1])
        Q = F.pad(new_Q, (0, 0, 0, 0, 0, 0, 1, 0))
        if run is not None:
            wandb.log(dict(rmse=rmse), step=step)
        if final:
            return
        if rmse <= rmse_bellman:
            final = True
