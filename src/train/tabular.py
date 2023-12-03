import math
import time
from collections import Counter
from dataclasses import asdict
from typing import Optional

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from wandb.sdk.wandb_run import Run

import wandb
from metrics import get_metrics
from models.tabular import DataPoint, SetTransformer
from train.make_tabular_data import make_data
from utils import decay_lr, load, save, set_seed


def train(
    decay_args: dict,
    load_path: str,
    lr: float,
    model_args: dict,
    n_batch: int,
    n_epochs: int,
    n_plot: int,
    bellman_delta: int,
    evaluate_args: dict,
    run: Optional[Run],
    save_interval: int,
    seed: int,
    test_1_interval: int,
    test_data_args: dict,
    test_n_interval: int,
    train_1_interval: int,
    train_data_args: dict,
    commit: str = None,
    config: str = None,
    config_name: str = None,
) -> None:
    del commit
    del config
    del config_name

    set_seed(seed)
    # create data

    kwargs = dict(bellman_delta=bellman_delta)
    train_data = make_data(**dict(**kwargs, seed=seed, **train_data_args))
    test_data = make_data(**dict(**kwargs, seed=seed + 1, **test_data_args))

    print("Create net... ", end="", flush=True)
    net = SetTransformer(
        **model_args,
        n_actions=train_data.n_actions,
        n_tokens=train_data.n_tokens,
    )
    if load_path is not None:
        load(load_path, net, run)
    net: SetTransformer = net.cuda()
    print("✓")

    counter = Counter()
    save_count = 0
    test_1_log = {}
    test_n_log = {}
    tick = time.time()
    if bellman_delta is None:
        bellman_delta = test_data.n_bellman_convergance
    iterations = int(math.ceil(test_data.n_bellman_convergance / bellman_delta))
    plot_indices = torch.randint(0, len(test_data), [n_plot]).cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    for e in range(n_epochs):
        # Split the dataset into train and test sets
        train_loader = DataLoader(train_data, batch_size=n_batch, shuffle=True)
        x: DataPoint
        for t, x in enumerate(train_loader):
            x = DataPoint(*[x.cuda() for x in x])
            step = e * len(train_loader) + t
            if step % test_1_interval == 0:
                log = test_data.evaluate(
                    iterations=1,
                    n_batch=n_batch,
                    net=net,
                    plot_indices=plot_indices,
                    **evaluate_args,
                )
                test_1_log = {f"test-1/{k}": v for k, v in log.items()}
            if step % test_n_interval == 0:
                log = test_data.evaluate(
                    iterations=iterations,
                    n_batch=n_batch,
                    net=net,
                    plot_indices=plot_indices,
                    **evaluate_args,
                )
                test_n_log = {f"test-n/{k}": v for k, v in log.items()}

            net.train()
            optimizer.zero_grad()

            outputs: torch.Tensor
            loss: torch.Tensor
            outputs, loss = net.forward(x)

            metrics = get_metrics(
                loss=loss,
                outputs=outputs,
                targets=x.target_q,
                **evaluate_args,
            )

            decayed_lr = decay_lr(lr, step=step, **decay_args)
            for param_group in optimizer.param_groups:
                param_group.update(lr=decayed_lr)

            loss.backward()
            optimizer.step()
            counter.update(asdict(metrics), n=1)
            if step % train_1_interval == 0:
                fps = train_1_interval / (time.time() - tick)
                tick = time.time()
                train_1_log = {
                    f"train-1/{k}": v / counter["n"] for k, v in counter.items()
                }
                train_1_log.update(
                    epoch=e, fps=fps, lr=decayed_lr, save_count=save_count
                )
                counter = Counter()
                print(".", end="", flush=True)
                if run is not None:
                    wandb.log(
                        dict(**test_1_log, **test_n_log, **train_1_log),
                        step=step,
                    )
                plt.close()
                test_1_log = {}
                test_n_log = filter_number(**test_n_log)

            # save
            if step % save_interval == 0:
                save(run, net)
                save_count += 1

    save(run, net)


def filter_number(**kwargs):
    return {k: v for k, v in kwargs.items() if isinstance(v, (float, int))}
