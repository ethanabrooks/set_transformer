import itertools
import math
import shutil
import time
from abc import abstractmethod
from dataclasses import asdict, dataclass
from typing import Counter, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

import wandb
from dataset.trajectories import Dataset
from envs.subproc_vec_env import SubprocVecEnv
from evaluate import log as log_evaluation
from evaluate import rollout
from metrics import Metrics, compute_rmse, get_metrics
from models.trajectories import Model
from sequence.base import Sequence
from utils import DataPoint, decay_lr, load, save
from values.bootstrap import Values as BootstrapValues


@dataclass
class Evaluator:
    envs: SubprocVecEnv
    max_iterations: Optional[int]
    net: Model
    rollout_length: int

    def rollout(self, iterations: int) -> pd.DataFrame:
        if self.max_iterations is not None:
            iterations = min(iterations, self.max_iterations)
        return rollout(
            envs=self.envs,
            iterations=iterations,
            net=self.net,
            rollout_length=self.rollout_length,
        )


@dataclass(frozen=True)
class Trainer:
    alpha: float
    baseline: bool
    bellman_delta: int
    count_threshold: int
    decay_args: dict
    evaluator: Evaluator
    load_path: Optional[str]
    log_interval: int
    metrics_args: dict
    n_batch: int
    net: Model
    optimizer: optim.Optimizer
    plot_indices: torch.Tensor
    rmse_bellman: float
    rmse_training_final: float
    rmse_training_intermediate: float
    run: Run
    sequence: Sequence
    test_interval: int

    @classmethod
    @abstractmethod
    def build_model(cls, **kwargs) -> nn.Module:
        pass

    @classmethod
    def make(
        cls,
        baseline: bool,
        bellman_delta: int,
        envs: SubprocVecEnv,
        evaluator_args: dict,
        load_path: Optional[str],
        lr: float,
        model_args: dict,
        n_plot: int,
        run: Run,
        sequence: Sequence,
        **kwargs,
    ):
        _, l = sequence.transitions.rewards.shape
        net = cls.build_model(
            bellman_delta=bellman_delta,
            **model_args,
            n_actions=sequence.n_actions,
            n_ctx=l,
            n_tokens=sequence.n_tokens,
            pad_value=sequence.pad_value,
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
        b, _ = sequence.transitions.rewards.shape
        plot_indices = torch.randint(0, b, (n_plot,))
        return cls(
            baseline=baseline,
            bellman_delta=bellman_delta,
            evaluator=evaluator,
            **kwargs,
            load_path=load_path,
            net=net,
            optimizer=optimizer,
            plot_indices=plot_indices,
            run=run,
            sequence=sequence,
        )

    def get_ground_truth(self, bellman_number: int) -> Optional[torch.Tensor]:
        pass

    def train(self, lr: float):
        b, l = self.sequence.transitions.rewards.shape
        a = self.sequence.n_actions
        Q = torch.zeros(1, b, l, a)
        final = self.baseline
        start_step = 0

        for bellman_number in itertools.count(1):
            new_Q, step = self.train_curriculum_stage(
                bellman_number=bellman_number,
                bootstrap_Q=Q,
                lr=lr,
                start_step=start_step,
                stop_at_rmse=self.rmse_training_final
                if final
                else self.rmse_training_intermediate,
            )
            start_step = step
            rmse = compute_rmse(Q[-1], new_Q[-1])
            Q = F.pad(new_Q, (0, 0, 0, 0, 0, 0, 1, 0))
            if self.run is not None:
                wandb.log(dict(rmse=rmse), step=step)
            if final:
                return
            if rmse <= self.rmse_bellman:
                final = True

    def train_curriculum_stage(
        self,
        bellman_number: int,
        bootstrap_Q: torch.Tensor,
        lr: float,
        start_step: int,
        stop_at_rmse: float,
    ):
        if self.run is not None:
            wandb.log(dict(bellman_number=bellman_number), step=start_step)

        assert torch.all(bootstrap_Q[0] == 0)
        sequence = self.sequence
        ground_truth = self.get_ground_truth(bellman_number)

        test_log = {}
        counter = Counter()
        Q = bootstrap_Q.clone()
        _, _, l, a = bootstrap_Q.shape
        updated = None

        def make_dataset(bootstrap_Q: torch.Tensor):
            assert len(bootstrap_Q) == bellman_number
            values = BootstrapValues.make(bootstrap_Q=bootstrap_Q, sequence=sequence)
            return Dataset(
                bellman_delta=self.bellman_delta, sequence=sequence, values=values
            )

        def _get_metrics(prefix: str, outputs: torch.Tensor, targets: torch.Tensor):
            metrics = get_metrics(
                loss=None, outputs=outputs, targets=targets, **self.metrics_args
            )
            return {f"{prefix}/{k}": v for k, v in asdict(metrics).items()}

        train_data = make_dataset(bootstrap_Q)
        tick = time.time()
        pbar = None

        for e in itertools.count():
            if updated is not None:
                assert torch.all(updated)
            updated = torch.zeros_like(Q)

            q, b, l, _ = Q.shape
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
            if ground_truth is not None and q <= len(ground_truth):
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

            if self.baseline:
                bootstrap_Q2 = Q
            else:
                bootstrap_Q2 = F.pad(Q, (0, 0, 0, 0, 0, 0, 1, 0))[:-1]
            bootstrap_Q = self.alpha * bootstrap_Q2 + (1 - self.alpha) * bootstrap_Q
            train_data = make_dataset(bootstrap_Q)
            train_loader = DataLoader(train_data, batch_size=self.n_batch, shuffle=True)
            epoch_step = start_step + e * len(train_loader)
            remaining = epoch_rmse - stop_at_rmse
            if pbar is None:
                pbar = tqdm(
                    total=round(remaining, 3),
                    desc=f"Stage {bellman_number} progress",
                    ncols=shutil.get_terminal_size().columns,
                )
            pbar.n = max(0, min(pbar.total, round(pbar.total - remaining, 3)))
            pbar.refresh()
            done = epoch_rmse <= stop_at_rmse
            if done:
                self.update_plots(bellman_number=bellman_number, Q=Q)
                save(self.run, self.net)
            if self.baseline:
                run_evaluation = e % self.test_interval == 0
            else:
                run_evaluation = done and (bellman_number % self.test_interval == 0)
            if bellman_number == 1 and e == 0:
                run_evaluation = True
            if self.evaluator is None:
                run_evaluation = False
            if run_evaluation:
                df = self.evaluator.rollout(
                    iterations=math.ceil(bellman_number / self.bellman_delta)
                )
                plot_log, test_log = log_evaluation(
                    count_threshold=self.count_threshold,
                    df=df,
                    run=self.run,
                    sequence=sequence,
                )
                if self.run is not None:
                    log = dict(
                        **plot_log, **{f"test/{k}": v for k, v in test_log.items()}
                    )
                    wandb.log(log, step=epoch_step)
            x: DataPoint
            self.net.train()
            for t, x in enumerate(train_loader):
                step = epoch_step + t
                decayed_lr = decay_lr(lr, step=step, **self.decay_args)
                if done or (step % self.log_interval == 0):
                    fps = self.log_interval / (time.time() - tick)
                    tick = time.time()
                    train_log = {k: v / counter["n"] for k, v in counter.items()}
                    test_log.update(bellman_number=bellman_number)
                    repeated_log = {f"repeat/{k}": v for k, v in test_log.items()}
                    self.update_plots(bellman_number=bellman_number, Q=Q)

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
                    if self.run is not None:
                        wandb.log(dict(**repeated_log, **train_log), step=step)
                if done:
                    return Q, epoch_step

                for param_group in self.optimizer.param_groups:
                    param_group.update(lr=decayed_lr)
                outputs, loss = self.net.forward_with_rotation(
                    x=x, optimizer=self.optimizer if self.load_path is None else None
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
                    **self.metrics_args,
                )
                counter.update(asdict(metrics), n=1)

    def update_plots(self, bellman_number: int, Q: torch.Tensor):
        pass
