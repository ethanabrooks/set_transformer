from typing import Optional
import tomli
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import logging
import time
import wandb
from models import SetTransformer, DeepSet
from mixture_of_mvns import MixtureOfMVNs
from mvn_diag import MultivariateNormalDiag
from rich.console import Console
from dollar_lambda import command

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
    lr: float = 1e-4,
    net: str = "set_transformer",
    notes: Optional[str] = None,
    num_steps: int = 50000,
    run_name: str = "trial",
    save_freq: int = 400,
    test_freq: int = 200,
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

    save_dir = os.path.join("results", net, run_name)
    if net == "set_transformer":
        console.log("B", B)
        console.log("K", K)
        console.log("S", S)
        console.log("D", D)

        net = SetTransformer(K, S, D).cuda()
        console.log("Input (B, K*S)", B, K * S)
        console.log("Output (B, S, D)", B, S, D)
    elif net == "deepset":
        net = DeepSet(K, S, D).cuda()
    else:
        raise ValueError("Invalid net {}".format(net))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(run_name)
    logger.addHandler(
        logging.FileHandler(
            os.path.join(save_dir, "train_" + time.strftime("%Y%m%d-%H%M") + ".log"),
            mode="w",
        )
    )
    ce_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    tick = time.time()
    for t in range(1, num_steps + 1):
        if t == int(0.5 * num_steps):
            optimizer.param_groups[0]["lr"] *= 0.1
        net.train()
        optimizer.zero_grad()
        N = np.random.randint(N_min, N_max)
        X = torch.randint(0, 2, (B, S)).cuda()
        Y = net(X)
        # console.log("X", X.shape)
        # console.log("Y", Y.shape)
        ll = ce_loss(Y, X)
        # I = torch.arange(B)[..., None]
        # logits_acc = torch.softmax(Y, -1)[I, X, :]
        argmax_acc = (Y.argmax(-1) == X).float()
        loss = ll
        if run is not None:
            wandb.log(
                {
                    "loss": loss.item(),
                    "ll": ll.item(),
                    "argmax_acc": argmax_acc.mean().item(),
                },
                step=t,
            )
        loss.backward()
        optimizer.step()

        if t % test_freq == 0:
            line = "step {}, lr {:.3e}, ".format(t, optimizer.param_groups[0]["lr"])
            # line += test(bench, verbose=False)
            line += " ({:.3f} secs)".format(time.time() - tick)
            tick = time.time()
            logger.info(line)

        if t % save_freq == 0:
            torch.save(
                {"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar")
            )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))


if __name__ == "__main__":
    main()
