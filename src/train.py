import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

import wandb
from dataset.base import Dataset
from metrics import get_metrics
from models.set_transformer import DataPoint, SetTransformer
from pretty import print_row
from sequence import make as make_sequence
from values import make as make_values

MODEL_FNAME = "model.tar"


def make_data(
    dataset_args: dict,
    sequence_args: dict,
    sample_from_trajectories: bool,
    seed: int,
) -> Dataset:
    sequence = make_sequence(
        sample_from_trajectories=sample_from_trajectories, seed=seed, **sequence_args
    )

    values = make_values(
        sequence=sequence, sample_from_trajectories=sample_from_trajectories
    )

    dataset: Dataset = Dataset.make(**dataset_args, sequence=sequence, values=values)
    return dataset


def load(
    load_path: Optional[str],
    net: SetTransformer,
    run: Optional[Run],
):
    root = run.dir if run is not None else f"/tmp/wandb{time.time()}"
    wandb.restore(MODEL_FNAME, run_path=load_path, root=root)
    state = torch.load(os.path.join(root, MODEL_FNAME))
    net.load_state_dict(state, strict=True)


def decay_lr(lr: float, final_step: int, step: int, warmup_steps: int):
    if step < warmup_steps:
        # linear warmup
        lr_mult = float(step) / float(max(1, warmup_steps))
    else:
        # cosine learning rate decay
        progress = float(step - warmup_steps) / float(max(1, final_step - warmup_steps))
        progress = np.clip(progress, 0.0, 1.0)
        lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr * lr_mult


def train(
    data_args: dict,
    decay_args: dict,
    load_path: str,
    lr: float,
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    n_plot: int,
    bellman_delta: int,
    evaluate_args: dict,
    run: Optional[Run],
    save_interval: int,
    seed: int,
    test_1_interval: int,
    test_data_args: dict,
    test_n_interval: int,
    train_1_interval: int,
    train_data_args: dict,
    commit: str = None,
    config: str = None,
    config_name: str = None,
) -> None:
    del commit
    del config
    del config_name

    # Set the seed for PyTorch
    torch.manual_seed(seed)

    # If you are using CUDA (GPU), you also need to set the seed for the CUDA device
    # This ensures reproducibility for GPU calculations as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Set the seed for NumPy
    np.random.seed(seed)

    # Set the seed for Python's random module
    random.seed(seed)

    # create data
    data_args = OmegaConf.create(data_args)

    def make_data_args(**kwargs: dict):
        kwargs = OmegaConf.create(kwargs)
        kwargs = OmegaConf.merge(data_args, kwargs)
        return kwargs

    train_data = make_data(**make_data_args(seed=seed, **train_data_args))
    test_data = make_data(**make_data_args(seed=seed + 1, **test_data_args))

    print("Create net... ", end="", flush=True)
    net = SetTransformer(
        **model_args, n_actions=train_data.n_actions, n_tokens=train_data.n_tokens
    )
    if load_path is not None:
        load(load_path, net, run)
    net: SetTransformer = net.cuda()
    print("✓")

    counter = Counter()
    save_count = 0
    test_1_log = {}
    test_n_log = {}
    tick = time.time()
    if bellman_delta is None:
        bellman_delta = test_data.max_n_bellman
    iterations = int(math.ceil(test_data.max_n_bellman / bellman_delta))
    plot_indices = torch.randint(0, len(test_data), [n_plot]).cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    for e in range(n_epochs):
        # Split the dataset into train and test sets
        train_loader = DataLoader(train_data, batch_size=n_batch, shuffle=True)
        for t, x in enumerate(train_loader):
            x = DataPoint(*[x.cuda() for x in x])
            step = e * len(train_loader) + t
            if step % test_1_interval == 0:
                log = test_data.evaluate(
                    bellman_delta=bellman_delta,
                    iterations=1,
                    n_batch=n_batch,
                    net=net,
                    plot_indices=plot_indices,
                    **evaluate_args,
                )
                test_1_log = {f"test-1/{k}": v for k, v in log.items()}
                print_row(test_1_log, show_header=True)
            if step % test_n_interval == 0:
                log = test_data.evaluate(
                    bellman_delta=bellman_delta,
                    iterations=iterations,
                    n_batch=n_batch,
                    net=net,
                    plot_indices=plot_indices,
                    **evaluate_args,
                )
                test_n_log = {f"test-n/{k}": v for k, v in log.items()}
                print_row(test_n_log, show_header=True)

            net.train()
            optimizer.zero_grad()

            values = train_data.index_values(x.values, x.input_bellman)
            q_values = train_data.index_values(
                x.q_values, x.input_bellman + bellman_delta
            )

            outputs: torch.Tensor
            loss: torch.Tensor
            outputs, loss = net.forward(x, values=values, q_values=q_values)

            metrics = get_metrics(
                loss=loss,
                outputs=outputs,
                targets=q_values,
                **evaluate_args,
            )

            decayed_lr = decay_lr(lr, step=step, **decay_args)
            for param_group in optimizer.param_groups:
                param_group.update(lr=decayed_lr)

            loss.backward()
            optimizer.step()
            counter.update(asdict(metrics), n=1)
            if step % train_1_interval == 0:
                fps = train_1_interval / (time.time() - tick)
                tick = time.time()
                train_1_log = {
                    f"train-1/{k}": v / counter["n"] for k, v in counter.items()
                }
                train_1_log.update(
                    epoch=e, fps=fps, lr=decayed_lr, save_count=save_count
                )
                counter = Counter()
                print_row(train_1_log, show_header=(step % train_1_interval == 0))
                if run is not None:
                    wandb.log(
                        dict(**test_1_log, **test_n_log, **train_1_log),
                        step=step,
                    )
                plt.close()
                test_1_log = {}
                test_n_log = filter_number(**test_n_log)

            # save
            if step % save_interval == 0:
                save(run, net)
                save_count += 1

    save(run, net)


def filter_number(**kwargs):
    return {k: v for k, v in kwargs.items() if isinstance(v, (float, int))}


def save(run: Run, net: SetTransformer):
    if run is not None:
        savepath = os.path.join(run.dir, MODEL_FNAME)
        torch.save(net.state_dict(), savepath)
        wandb.save(savepath)
