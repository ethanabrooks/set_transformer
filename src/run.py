import logging
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
    n_token: int = 200,
    notes: Optional[str] = None,
    num_steps: int = 100000,
    run_name: str = "trial",
    save_freq: int = 400,
    seq_len: int = 50,
    seq2seq: str = "gru",
    test_split: float = 0.02,
    test_freq: int = 500,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    run = (
        None
        if debug
        else wandb.init(
            config=vars(),
            notes=notes,
            project=project_name(),
        )
    )

    save_dir = os.path.join("results", run_name)
    console.log("B", n_batch)
    console.log("K", seq_len)

    net = SetTransformer(n_token, seq2seq=seq2seq).cuda()

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(level=logging.INFO)
    ce_loss = nn.CrossEntropyLoss()

    dataset = RLData(n_token, num_steps, seq_len)

    # Split the dataset into train and test sets
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Split the dataset into train and test sets
    train_loader = DataLoader(train_dataset, batch_size=n_batch, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=n_batch, shuffle=False)

    optimizer = optim.Adam(net.parameters(), lr=lr)
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

        if t == int(0.5 * num_steps):
            optimizer.param_groups[0]["lr"] *= 0.1
        net.train()
        optimizer.zero_grad()

        Y = net(X)
        # console.log("X", X.shape)
        # console.log("Y", Y.shape)
        loss = ce_loss(Y.swapaxes(1, 2), Z)
        assert [*Y.shape] == [n_batch, seq_len, n_token]
        # I = torch.arange(B)[..., None]
        # logits_acc = torch.softmax(Y, -1)[I, X, :]
        argmax_acc = (Y.argmax(-1) == Z).float().mean()
        loss.backward()
        optimizer.step()
        if t % log_freq == 0:
            log = dict(
                loss=loss.item(),
                argmax_acc=argmax_acc.item(),
            )

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
