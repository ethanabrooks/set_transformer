import itertools
import time
from dataclasses import asdict
from pathlib import Path
from typing import Counter, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

import wandb
from dataset.value_unconditional import DataPoint, Dataset
from metrics import Metrics, compute_rmse, get_metrics
from models.value_unconditional import SetTransformer
from pretty import print_row
from sequence import make as make_sequence
from sequence.base import Sequence
from utils import decay_lr, set_seed
from values.bootstrap import Values as BootstrapValues


def train_bellman_iteration(
    bellman_number: int,
    bootstrap_Q: torch.Tensor,
    decay_args: dict,
    evaluate_args: dict,
    log_interval: int,
    lr: float,
    optimizer: optim.Optimizer,
    plot_indices: torch.Tensor,
    sequence: Sequence,
    n_batch: int,
    net: SetTransformer,
    run: Run,
    start_step: int,
    stop_at_rmse: float,
    test_interval: int,
    test_size: int,
):
    # TODO: implement testing
    del test_interval
    del test_size

    assert torch.all(bootstrap_Q[0] == 0)
    ground_truth = sequence.grid_world.Q[
        1 : 1 + bellman_number  # omit Q_0 from ground_truth
    ]

    def make_dataset(bootstrap_Q: torch.Tensor):
        assert len(bootstrap_Q) == bellman_number
        values = BootstrapValues.make(
            bootstrap_Q=bootstrap_Q, sequence=sequence, stop_at_rmse=stop_at_rmse
        )
        return Dataset.make(sequence=sequence, values=values)

    test_log = {}
    tick = time.time()
    counter = Counter()

    Q = bootstrap_Q.clone()
    updated = None

    for e in itertools.count():
        q, b, l, _ = bootstrap_Q.shape
        bootstrap_Q = F.pad(Q, (0, 0, 0, 0, 0, 0, 1, 0)).cpu()[:-1]
        train_data = make_dataset(bootstrap_Q)
        train_loader = DataLoader(train_data, batch_size=n_batch, shuffle=True)
        epoch_step = start_step + e * len(train_loader)
        Q_k1 = train_data.values.Q
        Q_k = Q[
            torch.arange(q)[:, None, None],
            torch.arange(b)[None, :, None],
            torch.arange(l)[None, None],
            sequence.transitions.actions[None],
        ]
        epoch_rmse = compute_rmse(Q_k, Q_k1)
        if updated is not None:
            assert torch.all(updated)
        updated = torch.zeros_like(Q)
        if epoch_rmse <= stop_at_rmse:
            return Q.cpu(), epoch_step
        xs: list[torch.Tensor]
        for t, xs in enumerate(train_loader):
            x = DataPoint(*[x.cuda() for x in xs])
            step = epoch_step + t
            net.train()
            optimizer.zero_grad()
            outputs: torch.Tensor
            loss: torch.Tensor
            q_values = x.q_values[torch.arange(len(x.q_values)), x.n_bellman]
            outputs, loss = net.forward(x, q_values=q_values)

            idxs = x.n_bellman.cpu(), x.idx.cpu()
            Q[idxs] = outputs.detach().cpu()
            updated[idxs] = 1

            def _get_metrics(prefix: str, outputs: torch.Tensor, targets: torch.Tensor):
                metrics = get_metrics(
                    loss=None, outputs=outputs, targets=targets, **evaluate_args
                )
                return {f"{prefix}/{k}": v for k, v in asdict(metrics).items()}

            b, l, _ = outputs.shape
            metrics: Metrics = get_metrics(
                loss=loss,
                outputs=outputs[
                    torch.arange(b)[:, None], torch.arange(l)[None], x.actions
                ],
                targets=q_values,
                **evaluate_args,
            )

            if x.n_bellman.max() < len(ground_truth):
                idxs = x.n_bellman[:, None].cpu(), x.idx[:, None].cpu(), x.states.cpu()
                ground_truth_metrics = _get_metrics(
                    "(ground-truth)",
                    outputs=outputs,
                    targets=ground_truth[idxs].cuda(),
                )
                versus_metrics = _get_metrics(
                    "(botstrap versus ground-truth)",
                    outputs=q_values.cpu(),
                    targets=ground_truth[[*idxs, x.actions.cpu()]],
                )
            else:
                ground_truth_metrics = {}
                versus_metrics = {}

            mask = torch.isin(x.idx, plot_indices)
            if x.n_bellman.max() < len(ground_truth) and mask.any():
                stacked = torch.stack(
                    [
                        outputs.detach().cpu(),
                        ground_truth[
                            x.n_bellman[:, None].cpu(),
                            x.idx[:, None].cpu(),
                            x.states.cpu(),
                        ],
                    ],
                    1,
                )
                grid_world = sequence.grid_world
                values = stacked[..., :: grid_world.n_actions, :].swapaxes(0, 1)
                idxs = x.idx[mask].cpu()
                Pi = grid_world.Pi[idxs][None, None]
                plot_values = values[:, mask.cpu()]
                plot_values: torch.Tensor = plot_values * Pi
                plot_values = plot_values.sum(-1).cpu()
                plot_values = torch.unbind(plot_values, dim=2)
                for i, plot_value in zip(idxs, plot_values):
                    fig = grid_world.visualize_values(plot_value)
                    test_log[f"values-plot {i}"] = wandb.Image(fig)

            decayed_lr = decay_lr(lr, step=step, **decay_args)
            for param_group in optimizer.param_groups:
                param_group.update(lr=decayed_lr)
            loss.backward()
            optimizer.step()
            counter.update(asdict(metrics), n=1)
            if step % log_interval == 0:
                fps = log_interval / (time.time() - tick)
                tick = time.time()
                train_log = {k: v / counter["n"] for k, v in counter.items()}

                train_log.update(
                    bellman_number=bellman_number,
                    epoch=e,
                    epoch_rmse=epoch_rmse,
                    fps=fps,
                    **ground_truth_metrics,
                    lr=decayed_lr,
                    **versus_metrics,
                )
                train_log = {f"train-Q/{k}": v for k, v in train_log.items()}
                counter = Counter()
                print_row(train_log, show_header=(step % log_interval == 0))
                if run is not None:
                    wandb.log(dict(**train_log, **test_log), step=step)
                test_log = {}


def compute_values(
    lr: float,
    model_args: dict,
    n_plot: int,
    rmse_bellman: float,
    rmse_training_final: float,
    rmse_training_intermediate: float,
    run: Run,
    sequence: Sequence,
    test_size: int,
    train_args: dict,
    load_path: Optional[str] = None,
):
    B, L = sequence.transitions.rewards.shape
    A = sequence.grid_world.n_actions
    Q = torch.zeros(1, B, L, A)
    values = BootstrapValues.make(
        sequence=sequence, stop_at_rmse=rmse_bellman, bootstrap_Q=Q
    )
    data = Dataset.make(sequence=sequence, values=values)
    start_step = 0
    n_tokens = max(data.n_tokens, len(sequence.grid_world.Q) * 2)  # double for padding
    net = SetTransformer(
        **model_args, n_actions=data.n_actions, n_tokens=n_tokens
    ).cuda()
    if load_path is not None:
        raise NotImplementedError
    optimizer = optim.Adam(net.parameters(), lr=lr)
    plot_indices = torch.randint(0, B, (n_plot,)).cuda()
    final = False

    for bellman_number in itertools.count(1):
        new_Q, step = train_bellman_iteration(
            bellman_number=bellman_number,
            bootstrap_Q=Q,
            lr=lr,
            optimizer=optimizer,
            sequence=sequence,
            net=net,
            plot_indices=plot_indices,
            run=run,
            start_step=start_step,
            stop_at_rmse=rmse_training_final if final else rmse_training_intermediate,
            test_size=test_size,
            **train_args,
        )
        start_step = step
        rmse = compute_rmse(Q[-1], new_Q[-1])
        Q = F.pad(new_Q, (0, 0, 0, 0, 0, 0, 1, 0))
        if run is not None:
            wandb.log({"Q/rmse": rmse}, step=step)
        if final:
            if run is not None:
                artifact = wandb.Artifact(name=run.id, type="Q")
                path = Path(run.dir) / "Q.pt"
                torch.save(Q, path)
                artifact.add_file(path)
                run.log_artifact(artifact)
            return Q.cpu()
        if rmse <= rmse_bellman:
            final = True


def train(*args, seed: int, sequence_args: dict, **kwargs):
    set_seed(seed)
    sequence = make_sequence(**sequence_args)
    return compute_values(*args, **kwargs, sequence=sequence)
