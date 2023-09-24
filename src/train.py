import os
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from wandb.sdk.wandb_run import Run

import wandb
from data import RLData
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


@dataclass
class Metrics:
    loss: float
    accuracy: float
    within1accuracy: float
    within2accuracy: float


def get_metrics(loss_fn, outputs, targets) -> tuple[torch.Tensor, Metrics]:
    loss = loss_fn(outputs.swapaxes(1, 2), targets)
    accuracy = (outputs.argmax(-1) == targets).float().mean().item()
    within1accuracy = ((outputs.argmax(-1) - targets).abs() <= 1).float().mean().item()
    within2accuracy = ((outputs.argmax(-1) - targets).abs() <= 2).float().mean().item()
    metrics = Metrics(
        loss=loss.item(),
        accuracy=accuracy,
        within1accuracy=within1accuracy,
        within2accuracy=within2accuracy,
    )
    return loss, metrics


def evaluate(net: nn.Module, test_loader: DataLoader):
    net.eval()
    counter = Counter()
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, Z in test_loader:
            Y = net.forward(X)
            _, metrics = get_metrics(loss_fn, Y, Z)
            counter.update(asdict(metrics))
    return {f"eval/{k}": v / len(test_loader) for k, v in counter.items()}


def train(
    data_args: dict,
    load_path: str,
    log_freq: int,
    lr: float,
    min_layers: Optional[int],
    max_layers: Optional[int],
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    run: Optional[Run],
    save_freq: int,
    seed: int,
    test_split: float,
    test_freq: int,
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

    dataset = RLData(**data_args)

    print("Create net... ", end="", flush=True)
    n_tokens = dataset.X.max().item() + 1
    dim_output = dataset.Z.max().item() + 1
    net = SetTransformer(**model_args, dim_output=dim_output, n_tokens=n_tokens)
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

    optimizer = optim.Adam(net.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    for e in range(n_epochs):
        # Split the dataset into train and test sets
        train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=n_batch, shuffle=False)
        for t, (X, Z) in enumerate(train_loader):
            step = e * len(train_loader) + t
            if t % test_freq == 0:
                log = evaluate(net=net, test_loader=test_loader)
                print_row(log, show_header=True)
                if run is not None:
                    wandb.log(log, step=step)

            net.train()
            optimizer.zero_grad()

            Y = net.forward(X)

            loss, metrics = get_metrics(ce_loss, Y, Z)

            loss.backward()
            optimizer.step()
            counter.update(asdict(metrics))
            if t % log_freq == 0:
                log = {f"train/{k}": v / log_freq for k, v in counter.items()}
                log.update(save_count=save_count, epoch=e)
                counter = Counter()
                print_row(log, show_header=(t % test_freq == 0))
                if run is not None:
                    wandb.log(log, step=step)

                    # save
            if t % save_freq == 0:
                save(run, net)
                save_count += 1

    save(run, net)


def save(run: Run, net: SetTransformer):
    if run is not None:
        savepath = os.path.join(run.dir, MODEL_FNAME)
        torch.save(net.state_dict(), savepath)
        wandb.save(savepath)
