import tomli
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import logging
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb
from models import SetTransformer, DeepSet
from mixture_of_mvns import MixtureOfMVNs
from mvn_diag import MultivariateNormalDiag
from rich.console import Console

console = Console()

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--num_bench", type=int, default=100)
parser.add_argument("--net", type=str, default="set_transformer")
parser.add_argument("--B", type=int, default=10)
parser.add_argument("--N_min", type=int, default=300)
parser.add_argument("--N_max", type=int, default=600)
parser.add_argument("--K", type=int, default=4)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--run_name", type=str, default="trial")
parser.add_argument("--num_steps", type=int, default=50000)
parser.add_argument("--test_freq", type=int, default=200)
parser.add_argument("--save_freq", type=int, default=400)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--notes")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

B = args.B
N_min = args.N_min
N_max = args.N_max
S = args.K

K = 2
D = 2 * K
mvn = MultivariateNormalDiag(K)
mog = MixtureOfMVNs(mvn)


def project_name():
    with open("pyproject.toml", "rb") as f:
        pyproject = tomli.load(f)
    return pyproject["tool"]["poetry"]["name"]


run = (
    None
    if args.debug
    else wandb.init(
        config=vars(args),
        notes=args.notes,
        project=project_name(),
    )
)

if args.net == "set_transformer":
    console.log("B", B)
    console.log("K", K)
    console.log("S", S)
    console.log("D", D)

    net = SetTransformer(K, S, D).cuda()
    console.log("Input (B, K*S)", B, K * S)
    console.log("Output (B, S, D)", B, S, D)
elif args.net == "deepset":
    net = DeepSet(K, S, D).cuda()
else:
    raise ValueError("Invalid net {}".format(args.net))
benchfile = os.path.join("benchmark", "mog_{:d}.pkl".format(S))


save_dir = os.path.join("results", args.net, args.run_name)


def train():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(args.run_name)
    logger.addHandler(
        logging.FileHandler(
            os.path.join(save_dir, "train_" + time.strftime("%Y%m%d-%H%M") + ".log"),
            mode="w",
        )
    )
    logger.info(str(args) + "\n")
    ce_loss = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    tick = time.time()
    for t in range(1, args.num_steps + 1):
        if t == int(0.5 * args.num_steps):
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

        if t % args.test_freq == 0:
            line = "step {}, lr {:.3e}, ".format(t, optimizer.param_groups[0]["lr"])
            # line += test(bench, verbose=False)
            line += " ({:.3f} secs)".format(time.time() - tick)
            tick = time.time()
            logger.info(line)

        if t % args.save_freq == 0:
            torch.save(
                {"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar")
            )

    torch.save({"state_dict": net.state_dict()}, os.path.join(save_dir, "model.tar"))


if __name__ == "__main__":
    train()
