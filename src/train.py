import math
import os
import random
import time
from collections import Counter
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from wandb.sdk.wandb_run import Run

import wandb
from data.sample_uniform import RLData
from metrics import get_metrics
from models import SetTransformer
from pretty import print_row

MODEL_FNAME = "model.tar"


def load(
    load_path: Optional[str],
    net: SetTransformer,
    run: Optional[Run],
):
    root = run.dir if run is not None else f"/tmp/wandb{time.time()}"
    wandb.restore(MODEL_FNAME, run_path=load_path, root=root)
    state = torch.load(os.path.join(root, MODEL_FNAME))
    net.load_state_dict(state, strict=True)


def evaluate(
    iterations: int,
    net: nn.Module,
    bellman_delta: int,
    test_loader: DataLoader,
):
    net.eval()
    counter = Counter()
    with torch.no_grad():
        for input_n_bellman, action_probs, discrete, *values in test_loader:
            max_n_bellman = len(values) - 1
            v1 = values[0]
            final_outputs = torch.zeros_like(v1)
            for i in range(iterations):
                outputs: torch.Tensor
                loss: torch.Tensor
                outputs, loss = net.forward(
                    v1=v1,
                    action_probs=action_probs,
                    discrete=discrete,
                    targets=values[min((i + 1) * bellman_delta, max_n_bellman)],
                )
                v1 = outputs.squeeze(-1)
                mask = (input_n_bellman + i * bellman_delta) < max_n_bellman
                final_outputs[mask] = v1[mask]

            metrics = get_metrics(
                loss=loss,
                outputs=final_outputs,
                targets=values[iterations],
            )
            counter.update(asdict(metrics))
    return {k: v / len(test_loader) for k, v in counter.items()}


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
    loss: str,
    lr: float,
    max_test: int,
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    bellman_delta: int,
    run: Optional[Run],
    save_interval: int,
    seed: int,
    test_split: float,
    test_1_interval: int,
    test_n_interval: int,
    train_1_interval: int,
    train_n_interval: int,
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

    dataset = RLData(**data_args, seed=seed)

    print("Create net... ", end="", flush=True)
    n_tokens = dataset.discrete.max().item() + 1
    net = SetTransformer(**model_args, n_tokens=n_tokens)
    if load_path is not None:
        load(load_path, net, run)
    net = net.cuda()
    print("✓")

    # Split the dataset into train and test sets
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    test_size = min(test_size, max_test)
    discard_size = len(dataset) - test_size - train_size
    train_dataset, test_dataset, _ = random_split(
        dataset, [train_size, test_size, discard_size]
    )
    random_indices = torch.randint(0, len(train_dataset), [test_size])
    train_n_dataset = Subset(train_dataset, random_indices)

    counter = Counter()
    save_count = 0
    test_1_log = None
    test_n_log = None
    train_1_log = None
    train_n_log = None
    tick = time.time()
    iterations = int(math.ceil(dataset.max_n_bellman / bellman_delta))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    for e in range(n_epochs):
        # Split the dataset into train and test sets
        train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=n_batch, shuffle=False)
        train_n_loader = DataLoader(train_n_dataset, batch_size=n_batch, shuffle=False)
        for t, (_, action_probs, discrete, *values) in enumerate(train_loader):
            step = e * len(train_loader) + t
            if t % test_1_interval == 0:
                log = evaluate(
                    iterations=1,
                    net=net,
                    bellman_delta=bellman_delta,
                    test_loader=test_loader,
                )
                test_1_log = {f"test-1/{k}": v for k, v in log.items()}
                print_row(test_1_log, show_header=True)
            if t % test_n_interval == 0:
                log = evaluate(
                    iterations=iterations,
                    net=net,
                    bellman_delta=bellman_delta,
                    test_loader=test_loader,
                )
                test_n_log = {f"test-n/{k}": v for k, v in log.items()}
                print_row(test_n_log, show_header=True)

            if t % train_n_interval == 0:
                log = evaluate(
                    iterations=iterations,
                    net=net,
                    bellman_delta=bellman_delta,
                    test_loader=train_n_loader,
                )
                train_n_log = {f"train-n/{k}": v for k, v in log.items()}
                print_row(train_n_log, show_header=True)

            net.train()
            optimizer.zero_grad()

            targets_index = min(bellman_delta, dataset.max_n_bellman)
            outputs, loss = net.forward(
                v1=values[0],
                action_probs=action_probs,
                discrete=discrete,
                targets=values[targets_index],
            )

            metrics = get_metrics(
                loss=loss,
                outputs=outputs,
                targets=values[targets_index],
            )

            decayed_lr = decay_lr(lr, step=step, **decay_args)
            for param_group in optimizer.param_groups:
                param_group.update(lr=decayed_lr)

            loss.backward()
            optimizer.step()
            counter.update(asdict(metrics), n=1)
            if t % train_1_interval == 0:
                fps = train_1_interval / (time.time() - tick)
                tick = time.time()
                train_1_log = {
                    f"train/{k}": v / counter["n"] for k, v in counter.items()
                }
                train_1_log.update(
                    epoch=e, fps=fps, lr=decayed_lr, save_count=save_count
                )
                counter = Counter()
                print_row(train_1_log, show_header=(t % train_1_interval == 0))
                if run is not None:
                    wandb.log(
                        dict(**test_1_log, **test_n_log, **train_1_log, **train_n_log),
                        step=step,
                    )

                    # save
            if t % save_interval == 0:
                save(run, net)
                save_count += 1

    save(run, net)


def save(run: Run, net: SetTransformer):
    if run is not None:
        savepath = os.path.join(run.dir, MODEL_FNAME)
        torch.save(net.state_dict(), savepath)
        wandb.save(savepath)
