import os
import random
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


def train(
    log_freq: int,
    lr: float,
    n_batch: int,
    n_epochs: int,
    n_isab: int,
    n_sab: int,
    n_steps: int,
    run: Optional[Run],
    run_name: str,
    save_freq: int,
    seed: int,
    seq_len: int,
    seq2seq: str,
    test_split: float,
    test_freq: int,
    **data_args,
) -> None:
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
    net = SetTransformer(
        n_sab=n_sab,
        n_isab=n_isab,
        n_tokens=n_tokens,
        dim_output=dim_output,
        seq2seq=seq2seq,
    ).cuda()
    print("✓")

    # Split the dataset into train and test sets
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    optimizer = optim.Adam(net.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    for e in range(n_epochs):
        # Split the dataset into train and test sets
        train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=n_batch, shuffle=False)
        for t, (X, Z) in enumerate(train_loader):
            if t % test_freq == 0:
                net.eval()
                loss = 0
                argmax_acc = 0
                with torch.no_grad():
                    for X, Z in test_loader:
                        Y = net(X)
                        loss += ce_loss(Y.swapaxes(1, 2), Z)
                        argmax_acc += (Y.argmax(-1) == Z).float().mean()
                log = dict(loss=loss, argmax_acc=argmax_acc)
                log = {
                    f"test/{k}": (v / len(test_loader)).item() for k, v in log.items()
                }
                print_row(log, show_header=True)
                if run is not None:
                    wandb.log(log, step=e * len(train_loader) + t)

            if t == int(0.5 * n_steps):
                optimizer.param_groups[0]["lr"] *= 0.1
            net.train()
            optimizer.zero_grad()

            Y = net(X)
            # console.log("X", X.shape)
            # console.log("Y", Y.shape)
            loss = ce_loss(Y.swapaxes(1, 2), Z)
            assert [*Y.shape] == [n_batch, seq_len, dim_output]
            # I = torch.arange(B)[..., None]
            # logits_acc = torch.softmax(Y, -1)[I, X, :]
            argmax_acc = (Y.argmax(-1) == Z).float().mean()
            loss.backward()
            optimizer.step()
            if t % log_freq == 0:
                log = dict(loss=loss, argmax_acc=argmax_acc)
                log = {f"train/{k}": (v).item() for k, v in log.items()}

                print_row(log, show_header=(t % test_freq == 0))
                if run is not None:
                    wandb.log(log, step=e * len(train_loader) + t)

            if t % save_freq == 0:
                torch.save(
                    {"state_dict": net.state_dict()},
                    os.path.join(save_dir, "model.tar"),
                )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))
