import os
from typing import Optional

import tomli
import torch
import torch.nn as nn
import torch.optim as optim
from dollar_lambda import command
from rich.console import Console
from torch.utils.data import DataLoader, random_split

import wandb
from data import RLData
from models import SetTransformer
from pretty import print_row

console = Console()


def project_name():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["tool"]["poetry"]["name"]


@command()
def main(
    n_batch: int = 10,
    debug: bool = False,
    gpu: str = "0",
    log_freq: int = 20,
    lr: float = 1e-4,
    n_steps: int = 100000,
    grid_size: int = 5,
    notes: Optional[str] = None,
    run_name: str = "trial",
    save_freq: int = 400,
    seq_len: int = 100,
    seq2seq: str = "gru",
    test_split: float = 0.02,
    test_freq: int = 100,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    run = (
        None
        if debug
        else wandb.init(
            config=dict(
                n_batch=n_batch,
                debug=debug,
                gpu=gpu,
                log_freq=log_freq,
                lr=lr,
                n_steps=n_steps,
                grid_size=grid_size,
                notes=notes,
                run_name=run_name,
                save_freq=save_freq,
                seq_len=seq_len,
                seq2seq=seq2seq,
                test_split=test_split,
                test_freq=test_freq,
            ),
            notes=notes,
            project=project_name(),
        )
    )

    save_dir = os.path.join("results", run_name)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    dataset = RLData(grid_size, n_steps, seq_len)

    print("Create net... ", end="", flush=True)
    n_tokens = dataset.X.max().item() + 1
    dim_output = dataset.Z.max().item() + 1
    net = SetTransformer(
        n_tokens=n_tokens, dim_output=dim_output, seq2seq=seq2seq
    ).cuda()
    print("✓")

    # Split the dataset into train and test sets
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Split the dataset into train and test sets
    train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=n_batch, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
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
            log = {f"test/{k}": (v / len(test_loader)).item() for k, v in log.items()}
            print_row(log, show_header=True)
            if run is not None:
                wandb.log(log, step=t)

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
                wandb.log(log, step=t)

        if t % save_freq == 0:
            torch.save(
                {"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar")
            )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))


if __name__ == "__main__":
    main()
