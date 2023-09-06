import logging
import os
from typing import Optional

import tomli
import torch
import torch.nn as nn
import torch.optim as optim
from dollar_lambda import command
from rich.console import Console

import wandb
from models import SetTransformer
from pretty import print_row

console = Console()


def project_name():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["tool"]["poetry"]["name"]


@command()
def main(
    B: int = 10,
    K: int = 4,
    N_max: int = 600,
    N_min: int = 300,
    debug: bool = False,
    gpu: str = "0",
    log_freq: int = 20,
    lr: float = 1e-4,
    notes: Optional[str] = None,
    num_steps: int = 50000,
    run_name: str = "trial",
    save_freq: int = 400,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    B = B
    N_min = N_min
    N_max = N_max
    S = K

    K = 2
    D = 2 * K

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
    console.log("B", B)
    console.log("K", K)
    console.log("S", S)
    console.log("D", D)

    net = SetTransformer(K, D).cuda()
    console.log("Input (B, K*S)", B, K * S)
    console.log("Output (B, S, D)", B, S, D)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(level=logging.INFO)
    ce_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    for t in range(num_steps):
        if t == int(0.5 * num_steps):
            optimizer.param_groups[0]["lr"] *= 0.1
        net.train()
        optimizer.zero_grad()
        X = torch.randint(0, 2, (B, S)).cuda()
        Y = net(X)
        # console.log("X", X.shape)
        # console.log("Y", Y.shape)
        loss = ce_loss(Y, X)
        # I = torch.arange(B)[..., None]
        # logits_acc = torch.softmax(Y, -1)[I, X, :]
        argmax_acc = (Y.argmax(-1) == X).float()
        if t % log_freq == 0:
            log = dict(
                loss=loss.item(),
                argmax_acc=argmax_acc.mean().item(),
            )

            print_row(log, show_header=(t % (log_freq * 30) == 0))
            if run is not None:
                wandb.log(log, step=t)
        loss.backward()
        optimizer.step()

        if t % save_freq == 0:
            torch.save(
                {"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar")
            )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))


if __name__ == "__main__":
    main()
