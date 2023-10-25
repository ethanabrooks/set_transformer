import importlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import wandb
from metrics import get_metrics
from tabular.value_iteration import ValueIteration


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
        self.Pi = Pi.cuda()

        print("Policy evaluation...")
        V = torch.stack(grid_world.evaluate_policy_iteratively(Pi, stop_at_rmse))
        self.V = V
        self._max_n_bellman = len(V) - 1
        if max_initial_bellman is None:
            max_initial_bellman = self._max_n_bellman

        states, actions, next_states, rewards = self.collect_data(**kwargs, Pi=Pi)

        # sample n_bellman -- number of steps of policy evaluation
        self._input_bellman = input_bellman = torch.randint(0, max_initial_bellman, [B])
        n_bellman = torch.arange(len(V))[:, None] + input_bellman[None, :]

        n_bellman = torch.clamp(n_bellman, 0, self.max_n_bellman)
        action_probs = Pi[torch.arange(B)[:, None], states]
        # Use advanced indexing to get the desired tensor

        v_mesh, b_mesh = torch.meshgrid(
            torch.arange(len(V)), torch.arange(B), indexing="ij"
        )
        v_mesh = v_mesh + input_bellman[None, :]
        v_mesh = torch.clamp(v_mesh, 0, self.max_n_bellman)
        V_indexed = V[v_mesh, b_mesh]
        V_indexed = V_indexed[
            torch.arange(len(V))[:, None, None],
            torch.arange(B)[None, :, None],
            states[None],
        ]
        for b in range(10):
            for i, s in enumerate(states[b]):
                for nb in range(len(V)):
                    left = V_indexed[nb, b, i]
                    right = V[min(len(V) - 1, nb + input_bellman[b]), b, s]
                    assert torch.all(left == right), (nb, b, s, left, right)

        self._values = V_indexed
        self._continuous = torch.Tensor(action_probs)
        discrete = [
            states[..., None],
            actions[..., None],
            next_states[..., None],
            rewards[..., None],
        ]
        self._discrete = torch.cat(discrete, -1).long()

        self.perm = torch.rand(B, S * A).argsort(dim=1)

        if omit_states_actions > 0:
            self._values = self.shuffle(self._values, 2)[:, :, omit_states_actions:]
            self._continuous, self._discrete = [
                self.shuffle(x, 1)[:, omit_states_actions:]
                for x in [self._continuous, self._discrete]
            ]

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
            self.values[:, idx],
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
        values: torch.Tensor,
        **kwargs,
    ):
        A = len(self.grid_world.deltas)
        Pi: torch.Tensor = self.Pi[idxs]
        Pi = Pi.repeat_interleave(A, 1)
        if self.omit_states_actions > 0:
            Pi = self.shuffle(Pi, 1, idxs=idxs)
            Pi = Pi[:, self.omit_states_actions :]
        _, max_n_bellman, _ = values.shape
        max_n_bellman -= 1
        v1 = values[:, 0]
        final_outputs = torch.zeros_like(v1)
        all_outputs = []
        for j in range(iterations):
            outputs: torch.Tensor
            loss: torch.Tensor
            targets = values[:, min((j + 1) * bellman_delta, max_n_bellman)]
            with torch.no_grad():
                outputs, loss = net.forward(v1=v1, **kwargs, targets=targets)
            outputs = outputs.squeeze(-1)
            all_outputs.append(torch.stack([outputs, targets]))
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
        outputs = torch.stack(all_outputs)
        return metrics, outputs

    def evaluate(
        self, n_batch: int, net: nn.Module, plot_indices: torch.Tensor, **kwargs
    ):
        net.eval()
        counter = Counter()
        loader = DataLoader(self, batch_size=n_batch, shuffle=False)
        all_outputs = []
        all_idxs = []
        with torch.no_grad():
            x: torch.Tensor
            for x in loader:
                (idxs, input_n_bellman, action_probs, discrete, values) = [
                    x.cuda() for x in x
                ]
                metrics, outputs = self.get_n_metrics(
                    idxs=idxs,
                    input_n_bellman=input_n_bellman,
                    net=net,
                    values=values,
                    action_probs=action_probs,
                    discrete=discrete,
                    **kwargs,
                )
                counter.update(metrics)
                all_outputs.append(outputs)
                all_idxs.append(idxs)
        metrics = {k: v / len(loader) for k, v in counter.items()}
        A = len(self.grid_world.deltas)

        # add values plots to metrics
        outputs = torch.cat(all_outputs, 2)
        idxs = torch.cat(all_idxs)
        values: torch.Tensor
        values = outputs[:, :, :, ::A]
        mask = torch.isin(idxs, plot_indices)
        idxs = idxs[mask].cpu()
        plot_values = values[:, :, mask]
        plot_values = plot_values.cpu()
        plot_values = torch.unbind(plot_values, dim=2)
        for i, plot_value in zip(idxs, plot_values):
            fig = self.grid_world.visualize_values(plot_value)
            metrics[f"values-plot {i}"] = wandb.Image(fig)

        return metrics

    def shuffle(self, x: torch.Tensor, i: int, idxs: Optional[torch.Tensor] = None):
        p = self.perm.to(x.device)
        if idxs is not None:
            p = p[idxs]
        for _ in range(i - 1):
            p = p[None]
        while p.dim() < x.dim():
            p = p[..., None]

        return torch.gather(x, i, p.expand_as(x))


def make(path: "str | Path", *args, **kwargs) -> RLData:
    path = Path(path)
    name = path.stem
    name = ".".join(path.parts)
    module = importlib.import_module(name)
    data: RLData = module.RLData(*args, **kwargs)
    return data
