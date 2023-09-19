import os
import random
from collections import Counter
from dataclasses import dataclass
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


@dataclass
class Metrics:
    loss: float
    accuracy: float

    def items_dict(self):
        d = dict(loss=self.loss, accuracy=self.accuracy)
        return {k: v.detach().cpu().item() for k, v in d.items()}


def get_metrics(loss_fn, outputs, targets):
    loss = loss_fn(outputs.swapaxes(1, 2), targets)
    accuracy = (outputs.argmax(-1) == targets).float().mean()
    return Metrics(loss=loss, accuracy=accuracy)


def evaluate(net: nn.Module, test_loader: DataLoader):
    net.eval()
    counter = Counter()
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, Z in test_loader:
            Y = net.forward(X)
            metrics = get_metrics(loss_fn, Y, Z)
            counter.update(metrics.items_dict())
    log = {k: v / len(test_loader) for k, v in counter.items()}
    return log


def train(
    data_args: dict,
    log_freq: int,
    lr: float,
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    n_steps: int,
    run: Optional[Run],
    run_name: str,
    save_freq: int,
    seed: int,
    seq_len: int,
    test_split: float,
    test_freq: int,
) -> None:
    n_isab = model_args["n_isab"]
    n_sab = model_args["n_sab"]
    if n_isab + n_sab > 7:
        exit(0)
    if n_isab + n_sab < 4:
        exit(0)
    save_dir = os.path.join("results", run_name)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

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

    dataset = RLData(**data_args, n_steps=n_steps, seq_len=seq_len)

    print("Create net... ", end="", flush=True)
    n_tokens = dataset.X.max().item() + 1
    dim_output = dataset.Z.max().item() + 1
    net = SetTransformer(**model_args, dim_output=dim_output, n_tokens=n_tokens).cuda()
    print("✓")

    # Split the dataset into train and test sets
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    counter = Counter()

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

            if t == int(0.5 * n_steps):
                optimizer.param_groups[0]["lr"] *= 0.1
            net.train()
            optimizer.zero_grad()

            Y = net.forward(X)

            metrics = get_metrics(ce_loss, Y, Z)
            metrics.loss.backward()
            optimizer.step()
            counter.update(metrics.items_dict())
            if t % log_freq == 0:
                log = {f"train/{k}": v / log_freq for k, v in counter.items()}
                counter = Counter()
                print_row(log, show_header=(t % test_freq == 0))
                if run is not None:
                    wandb.log(log, step=step)

            if t % save_freq == 0:
                torch.save(
                    {"state_dict": net.state_dict()},
                    os.path.join(save_dir, "model.tar"),
                )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))
