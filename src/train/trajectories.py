import itertools
import pickle
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
from dataset.trajectories import Dataset
from metrics import Metrics, compute_rmse, get_metrics
from models.trajectories import CausalTransformer, SetTransformer
from sequence import make as make_sequence
from sequence.base import Sequence
from utils import DataPoint, decay_lr, load, save, set_seed
from values.bootstrap import Values as BootstrapValues


def train_bellman_iteration(
    alpha: float,
    bellman_delta: int,
    bellman_number: int,
    bootstrap_Q: torch.Tensor,
    decay_args: dict,
    evaluate_args: dict,
    load_path: str,
    log_interval: int,
    lr: float,
    optimizer: optim.Optimizer,
    plot_indices: torch.Tensor,
    sequence: Sequence,
    n_batch: int,
    net: SetTransformer,
    run: Run,
    sample_from_trajectories: bool,
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

    test_log = {}
    counter = Counter()
    Q = bootstrap_Q.clone()
    _, _, l, _ = bootstrap_Q.shape
    updated = None

    def make_dataset(bootstrap_Q: torch.Tensor):
        assert len(bootstrap_Q) == bellman_number
        values = BootstrapValues.make(
            bootstrap_Q=bootstrap_Q,
            sample_from_trajectories=sample_from_trajectories,
            sequence=sequence,
            stop_at_rmse=stop_at_rmse,
        )
        return Dataset(bellman_delta=bellman_delta, sequence=sequence, values=values)

    def _get_metrics(prefix: str, outputs: torch.Tensor, targets: torch.Tensor):
        metrics = get_metrics(
            loss=None, outputs=outputs, targets=targets, **evaluate_args
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
            test_log[f"plot {i}, bellman {bellman_number}"] = wandb.Image(fig)

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

        bootstrap_Q2 = F.pad(Q, (0, 0, 0, 0, 0, 0, 1, 0))[:-1]
        bootstrap_Q = alpha * bootstrap_Q2 + (1 - alpha) * bootstrap_Q
        train_data = make_dataset(bootstrap_Q)
        train_loader = DataLoader(train_data, batch_size=n_batch, shuffle=True)
        epoch_step = start_step + e * len(train_loader)
        if epoch_rmse <= stop_at_rmse:
            update_plots()
            save(run, net)
            return Q, epoch_step
        x: DataPoint
        net.train()
        for t, x in enumerate(train_loader):
            step = epoch_step + t
            decayed_lr = decay_lr(lr, step=step, **decay_args)
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
                **evaluate_args,
            )
            counter.update(asdict(metrics), n=1)
            if step % log_interval == 0:
                fps = log_interval / (time.time() - tick)
                tick = time.time()
                train_log = {k: v / counter["n"] for k, v in counter.items()}
                update_plots()

                train_log.update(
                    bellman_number=bellman_number,
                    epoch=e,
                    epoch_rmse=epoch_rmse,
                    fps=fps,
                    **ground_truth_metrics,
                    lr=decayed_lr,
                    max_Q=train_data.values.Q.max().item(),
                    **versus_metrics,
                )
                train_log = {f"train-Q/{k}": v for k, v in train_log.items()}
                counter = Counter()
                print(".", end="", flush=True)
                if run is not None:
                    wandb.log(dict(**train_log, **test_log), step=step)

                test_log = {}


def compute_values(
    bellman_delta: int,
    lr: float,
    model_args: dict,
    model_type: str,
    n_plot: int,
    partial_observation: bool,
    rmse_bellman: float,
    rmse_training_final: float,
    rmse_training_intermediate: float,
    run: Run,
    sample_from_trajectories: bool,
    sequence: Sequence,
    test_size: int,
    train_args: dict,
    load_path: Optional[str] = None,
):
    B, L = sequence.transitions.rewards.shape
    A = sequence.grid_world.n_actions
    Q = torch.zeros(1, B, L, A)
    values = BootstrapValues.make(
        sample_from_trajectories=sample_from_trajectories,
        sequence=sequence,
        stop_at_rmse=rmse_bellman,
        bootstrap_Q=Q,
    )
    data = Dataset(bellman_delta=bellman_delta, sequence=sequence, values=values)
    start_step = 0
    n_tokens = max(data.n_tokens, len(sequence.grid_world.Q) * 2)  # double for padding
    if model_type == "gpt2":
        net = CausalTransformer(
            bellman_delta=bellman_delta,
            **model_args,
            n_actions=data.n_actions,
            n_ctx=L,
            n_tokens=n_tokens,
            partial_observation=partial_observation,
        )
    elif model_type == "set":
        net = SetTransformer(
            bellman_delta=bellman_delta,
            **model_args,
            n_actions=data.n_actions,
            n_tokens=n_tokens,
        )
    else:
        raise ValueError(f"Unknown model_type {model_type}")
    if load_path is not None:
        load(load_path, net, run)
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)
    plot_indices = torch.randint(0, B, (n_plot,))
    final = False

    for bellman_number in itertools.count(1):
        new_Q, step = train_bellman_iteration(
            bellman_delta=bellman_delta,
            bellman_number=bellman_number,
            bootstrap_Q=Q,
            load_path=load_path,
            lr=lr,
            optimizer=optimizer,
            sequence=sequence,
            net=net,
            plot_indices=plot_indices,
            run=run,
            sample_from_trajectories=sample_from_trajectories,
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
                path = Path(run.dir) / "Q.pt"
                torch.save(Q, path)
                save_artifact(path=path, run=run, type="Q")
            return Q
        if rmse <= rmse_bellman:
            final = True


def save_artifact(path: Path, run: Run, type: str):
    artifact = wandb.Artifact(name=f"{type}-{run.id}", type=type)
    artifact.add_file(path)
    run.log_artifact(artifact)


def train(
    *args,
    partial_observation: bool,
    run: Run,
    seed: int,
    sample_from_trajectories: bool,
    sequence_args: dict,
    **kwargs,
):
    set_seed(seed)
    sequence = make_sequence(
        partial_observation=partial_observation,
        **sequence_args,
        sample_from_trajectories=sample_from_trajectories,
    )
    if run is not None:
        path = Path(run.dir) / "sequence.pkl"
        with path.open("wb") as f:
            pickle.dump(sequence, f)
        save_artifact(path=path, run=run, type="sequence")
    return compute_values(
        *args,
        **kwargs,
        partial_observation=partial_observation,
        run=run,
        sample_from_trajectories=sample_from_trajectories,
        sequence=sequence,
    )
