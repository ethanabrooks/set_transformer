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
from torch.utils.data import DataLoader, random_split
from wandb.sdk.wandb_run import Run

import wandb
from data import RLData
from metrics import LossType, get_metrics
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
    decode_outputs: torch.Tensor,
    iterations: int,
    loss_type: LossType,
    net: nn.Module,
    order_delta: int,
    test_loader: DataLoader,
):
    net.eval()
    counter = Counter()
    with torch.no_grad():
        for action_probs, discrete, *values in test_loader:
            v1 = values[0]
            for i in range(1, iterations + 1):
                outputs: torch.Tensor
                loss: torch.Tensor
                outputs, loss = net.forward(
                    v1=v1,
                    action_probs=action_probs,
                    discrete=discrete,
                    targets=values[min(i * order_delta, len(values) - 1)],
                )
                v1 = outputs.squeeze(-1)
            metrics = get_metrics(
                decode_outputs=decode_outputs,
                loss=loss,
                loss_type=loss_type,
                outputs=outputs,
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
    log_interval: int,
    loss: str,
    lr: float,
    min_layers: Optional[int],
    max_layers: Optional[int],
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    order_delta: int,
    run: Optional[Run],
    save_interval: int,
    seed: int,
    test_split: float,
    test_1_interval: int,
    test_n_interval: int,
    commit: str = None,
    config: str = None,
    config_name: str = None,
) -> None:
    del commit
    del config
    del config_name

    def check_layers(n_isab: int, n_sab: int, **_):  # dead: disable
        del _
        n_layers = n_sab + n_isab
        if n_layers < (min_layers or 1):
            exit(0)
        if max_layers is not None and n_layers > max_layers:
            exit(0)

    check_layers(**model_args)

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

    loss_type = LossType[loss.upper()]

    dataset = RLData(**data_args, loss_type=loss_type)

    print("Create net... ", end="", flush=True)
    n_tokens = dataset.discrete.max().item() + 1
    dim_output = dataset.V.max().item() + 1
    net = SetTransformer(
        **model_args,
        n_output=dim_output,
        n_tokens=n_tokens,
        loss_type=loss_type,
    )
    if load_path is not None:
        load(load_path, net, run)
    net = net.cuda()
    print("✓")

    # Split the dataset into train and test sets
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    counter = Counter()
    save_count = 0
    test_1_log = None
    test_n_log = None
    tick = time.time()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    for e in range(n_epochs):
        # Split the dataset into train and test sets
        train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=n_batch, shuffle=False)
        for t, (action_probs, discrete, *values) in enumerate(train_loader):
            step = e * len(train_loader) + t
            if t % test_1_interval == 0:
                log = evaluate(
                    decode_outputs=dataset.decode_outputs,
                    iterations=1,
                    loss_type=loss_type,
                    net=net,
                    order_delta=order_delta,
                    test_loader=test_loader,
                )
                test_1_log = {f"test-1/{k}": v for k, v in log.items()}
                print_row(test_1_log, show_header=True)
            if t % test_n_interval == 0:
                iterations = int(math.ceil(dataset.max_order / order_delta))
                log = evaluate(
                    decode_outputs=dataset.decode_outputs,
                    iterations=iterations,
                    loss_type=loss_type,
                    net=net,
                    order_delta=order_delta,
                    test_loader=test_loader,
                )
                test_n_log = {f"test-n/{k}": v for k, v in log.items()}
                print_row(test_n_log, show_header=True)

            net.train()
            optimizer.zero_grad()

            targets_index = min(order_delta, dataset.max_order)
            outputs, loss = net.forward(
                v1=values[0],
                action_probs=action_probs,
                discrete=discrete,
                targets=values[targets_index],
            )
            # wrong = Y.argmax(-1) != Z
            # if wrong.any():
            #     idx = wrong.nonzero()
            #     _X = dataset.decode_inputs(X.cpu())
            #     _Z = dataset.decode_outputs(Z.cpu())
            #     _Y = dataset.decode_outputs(Y.argmax(-1).cpu())
            #     breakpoint()

            metrics = get_metrics(
                decode_outputs=dataset.decode_outputs,
                loss=loss,
                loss_type=loss_type,
                outputs=outputs,
                targets=values[targets_index],
            )

            decayed_lr = decay_lr(lr, step=step, **decay_args)
            for param_group in optimizer.param_groups:
                param_group.update(lr=decayed_lr)

            loss.backward()
            optimizer.step()
            counter.update(asdict(metrics))
            if t % log_interval == 0:
                fps = log_interval / (time.time() - tick)
                tick = time.time()
                train_log = {f"train/{k}": v / log_interval for k, v in counter.items()}
                train_log.update(epoch=e, fps=fps, lr=decayed_lr, save_count=save_count)
                counter = Counter()
                print_row(train_log, show_header=(t % test_1_interval == 0))
                if run is not None:
                    wandb.log(dict(**train_log, **test_1_log, **test_n_log), step=step)

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
