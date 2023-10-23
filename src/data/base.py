import importlib
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional, TypeVar

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import wandb
from metrics import get_metrics
from tabular.value_iteration import ValueIteration

T = TypeVar("T")


class RLData(Dataset, ABC):
    def __init__(
        self,
        grid_world_args: dict,
        max_initial_bellman: Optional[int],
        n_data: int,
        omit_states_actions: int,
        seed: int,
        stop_at_rmse: float,
        **kwargs,
    ):
        self.n_data = n_data
        self.omit_states_actions = omit_states_actions
        self.stop_at_rmse = stop_at_rmse
        # 2D deltas for up, down, left, right
        grid_world = ValueIteration(**grid_world_args, n_tasks=n_data, seed=seed)
        self.grid_world = grid_world
        A = len(grid_world.deltas)
        S = grid_world.n_states
        B = n_data

        alpha = torch.ones(A)
        Pi = torch.distributions.Dirichlet(alpha).sample((B, S))  # random policies
        assert [*Pi.shape] == [B, S, A]

        print("Policy evaluation...")
        V = torch.stack(grid_world.evaluate_policy_iteratively(Pi, stop_at_rmse))
        self.V = V
        self._max_n_bellman = len(V) - 1
        if max_initial_bellman is None:
            max_initial_bellman = self._max_n_bellman

        states, actions, next_states, rewards = self.collect_data(**kwargs, Pi=Pi)
        _, L = states.shape

        # sample n_bellman -- number of steps of policy evaluation
        self._input_bellman = input_bellman = torch.randint(
            0, max_initial_bellman, (B, 1)
        ).tile(1, L)

        n_bellman = [input_bellman + o for o in range(len(V))]
        n_bellman = [torch.clamp(o, 0, self.max_n_bellman) for o in n_bellman]
        arange = torch.arange(B)[:, None]
        action_probs = Pi[arange, states]
        V = [V[o, arange, states] for o in n_bellman]

        self._values = [torch.Tensor(v) for v in V]
        self._continuous = torch.Tensor(action_probs)
        discrete = [
            states[..., None],
            actions[..., None],
            next_states[..., None],
            rewards[..., None],
        ]
        self._discrete = torch.cat(discrete, -1).long()

        perm = torch.rand(B, S * A).argsort(dim=1)

        def shuffle(x: torch.Tensor):
            p = perm
            while p.dim() < x.dim():
                p = p[..., None]

            return torch.gather(x, 1, p.expand_as(x))

        if omit_states_actions > 0:
            *self._values, self._continuous, self._discrete = [
                shuffle(x)[:, omit_states_actions:]
                for x in [*self._values, self._continuous, self._discrete]
            ]
            self._input_bellman = input_bellman[:, omit_states_actions:].cuda()

        self.optimally_improved_policy_values = self.compute_improved_policy_value(
            self.V[-1]
        ).cuda()

    def __len__(self):
        return len(self.discrete)

    def __getitem__(self, idx):
        return (
            idx,
            self.input_bellman[idx],
            self.continuous[idx],
            self.discrete[idx],
            *[v[idx] for v in self.values],
        )

    @property
    def continuous(self) -> torch.Tensor:
        return self._continuous

    @property
    def discrete(self) -> torch.Tensor:
        return self._discrete

    @property
    def input_bellman(self) -> torch.Tensor:
        return self._input_bellman

    @property
    def max_n_bellman(self):
        return self._max_n_bellman

    @property
    def values(self) -> list[torch.Tensor]:
        return self._values

    @abstractmethod
    def collect_data(self):
        raise NotImplementedError

    def compute_improved_policy_value(
        self, values: torch.Tensor, idxs: Optional[torch.Tensor] = None
    ):
        Pi = self.grid_world.improve_policy(values, idxs=idxs)
        *_, values = self.grid_world.evaluate_policy_iteratively(
            Pi, self.stop_at_rmse, idxs=idxs
        )
        return values

    def get_n_metrics(
        self,
        accuracy_threshold: float,
        bellman_delta: int,
        idxs: torch.Tensor,
        input_n_bellman: torch.Tensor,
        iterations: int,
        net: nn.Module,
        plot_indices: torch.Tensor,
        values: torch.Tensor,
        **kwargs,
    ):
        max_n_bellman = len(values) - 1
        v1 = values[0]
        final_outputs = torch.zeros_like(v1)
        plot_mask = torch.isin(idxs, plot_indices)
        plot_values = defaultdict(list)
        for j in range(iterations):
            outputs: torch.Tensor
            loss: torch.Tensor
            targets = values[min((j + 1) * bellman_delta, max_n_bellman)]
            with torch.no_grad():
                outputs, loss = net.forward(v1=v1, **kwargs, targets=targets)
            outputs = outputs.squeeze(-1)
            for idx, output, target in zip(
                idxs[:, None][plot_mask], outputs[plot_mask], targets[plot_mask]
            ):
                plot_values[idx.item()].append([output, target])
            v1 = outputs
            mask = (input_n_bellman + j * bellman_delta) < max_n_bellman
            final_outputs[mask] = v1[mask]

        metrics = get_metrics(
            loss=loss,
            outputs=outputs,
            targets=targets,
            accuracy_threshold=accuracy_threshold,
        )
        metrics = asdict(metrics)
        return metrics, plot_values, outputs

    def evaluate(self, n_batch: int, net: nn.Module, **kwargs):
        net.eval()
        counter = Counter()
        loader = DataLoader(self, batch_size=n_batch, shuffle=False)
        plot_values = defaultdict(list)
        with torch.no_grad():
            x: torch.Tensor
            for x in loader:
                (idxs, input_n_bellman, action_probs, discrete, *values) = [
                    x.cuda() for x in x
                ]
                metrics, new_plot_values, _ = self.get_n_metrics(
                    idxs=idxs,
                    input_n_bellman=input_n_bellman,
                    net=net,
                    values=values,
                    action_probs=action_probs,
                    discrete=discrete,
                    **kwargs,
                )
                counter.update(metrics)
                for k, v in new_plot_values.items():
                    plot_values[k].extend(v)
        metrics = {k: v / len(loader) for k, v in counter.items()}
        for i, values in plot_values.items():
            values = torch.stack([torch.stack(v) for v in values]).cpu()
            fig = self.grid_world.visualize_values(
                values[..., :: len(self.grid_world.deltas)]
            )
            metrics[f"values-plot {i}"] = wandb.Image(fig)

        return metrics


def make(path: "str | Path", *args, **kwargs) -> RLData:
    path = Path(path)
    name = path.stem
    name = ".".join(path.parts)
    module = importlib.import_module(name)
    data: RLData = module.RLData(*args, **kwargs)
    return data
