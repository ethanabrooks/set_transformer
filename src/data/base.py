import importlib
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader

import wandb
from data.utils import Transition
from metrics import get_metrics
from tabular.grid_world import GridWorld


@dataclass(frozen=True)
class MDP(ABC):
    grid_world: GridWorld
    Pi: torch.Tensor
    transitions: Transition[torch.Tensor]

    @classmethod
    @abstractmethod
    def collect_data(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def make(
        cls,
        grid_world_args: dict,
        n_data: int,
        seed: int,
        **kwargs,
    ):
        # 2D deltas for up, down, left, right
        grid_world = GridWorld(**grid_world_args, n_tasks=n_data, seed=seed)
        A = len(grid_world.deltas)
        S = grid_world.n_states
        B = n_data

        alpha = torch.ones(A)
        Pi: torch.Tensor = torch.distributions.Dirichlet(alpha).sample(
            (B, S)
        )  # random policies
        assert [*Pi.shape] == [B, S, A]

        print("Policy evaluation...")
        # states, actions, next_states, rewards = self.collect_data(**kwargs, Pi=Pi)
        transitions: Transition[torch.Tensor] = cls.collect_data(
            **kwargs, grid_world=grid_world, Pi=Pi
        )
        return cls(
            grid_world=grid_world,
            Pi=Pi,
            transitions=transitions,
        )


class Dataset(torch.utils.data.Dataset, ABC):
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

        self.mdp_data = mdp_data = self.make_mdp(
            grid_world_args=grid_world_args, n_data=n_data, seed=seed, **kwargs
        )

        transitions = mdp_data.transitions
        states = transitions.states
        action_probs = transitions.action_probs
        self._action_probs = action_probs.cuda()

        B = n_data
        S = mdp_data.grid_world.n_states
        A = len(mdp_data.grid_world.deltas)
        Q = torch.stack(
            mdp_data.grid_world.evaluate_policy_iteratively(
                Pi=mdp_data.Pi, stop_at_rmse=stop_at_rmse
            )
        )
        self.Q = Q

        # sample n_bellman -- number of steps of policy evaluation
        self._max_n_bellman = len(Q) - 1
        if max_initial_bellman is None:
            max_initial_bellman = self._max_n_bellman
        self._input_bellman = input_bellman = torch.randint(0, max_initial_bellman, [B])
        n_bellman = torch.arange(len(Q))[:, None] + input_bellman[None, :]
        n_bellman = torch.clamp(n_bellman, 0, len(Q) - 1)

        # create Q_indexed/self.q_values
        q_mesh, b_mesh = torch.meshgrid(
            torch.arange(len(Q)), torch.arange(B), indexing="ij"
        )
        q_mesh = q_mesh + input_bellman[None, :]
        q_mesh = torch.clamp(q_mesh, 0, len(Q) - 1)
        Q_indexed = Q[q_mesh, b_mesh]
        Q_indexed = Q_indexed[
            torch.arange(len(Q))[:, None, None],
            torch.arange(B)[None, :, None],
            states[None],
        ]
        for b in range(10):
            for i, s in enumerate(states[b]):
                for nb in range(len(Q)):
                    left = Q_indexed[nb, b, i]
                    right = Q[min(len(Q) - 1, nb + input_bellman[b]), b, s]
                    assert torch.all(left == right), (nb, b, s, left, right)
        self._q_values = Q_indexed

        self.V = (Q * mdp_data.Pi[None]).sum(-1)
        V_indexed = self.V[q_mesh, b_mesh]
        V_indexed = V_indexed[
            torch.arange(len(Q))[:, None, None],
            torch.arange(B)[None, :, None],
            states[None],
        ]
        self._values = V_indexed

        self._continuous = torch.Tensor(action_probs)
        discrete = [
            transitions.states[..., None],
            transitions.actions[..., None],
            transitions.next_states[..., None],
            transitions.rewards[..., None],
        ]
        self._discrete = torch.cat(discrete, -1).long()

        self.perm = torch.rand(B, S * A).argsort(dim=1)

        if omit_states_actions > 0:
            self._q_values, self._values = [
                self.shuffle(x, 2)[:, :, omit_states_actions:]
                for x in [self._q_values, self._values]
            ]
            self._continuous, self._discrete, self._action_probs = [
                self.shuffle(x, 1)[:, omit_states_actions:]
                for x in [self._continuous, self._discrete, self._action_probs]
            ]

        self.optimally_improved_policy_values = self.compute_improved_policy_value(
            self.Q[-1]
        ).cuda()

    @abstractmethod
    def make_mdp(self, *args, **kwargs) -> MDP:
        raise NotImplementedError

    @property
    def discrete(self) -> torch.Tensor:
        return self._discrete

    @property
    def max_n_bellman(self):
        return self._max_n_bellman

    @property
    def n_actions(self):
        return len(self.mdp_data.grid_world.deltas)

    def __getitem__(self, idx):
        return (
            idx,
            self._input_bellman[idx],
            self._continuous[idx],
            self._discrete[idx],
            self._q_values[:, idx],
            self._values[:, idx],
        )

    def __len__(self):
        return len(self.discrete)

    def compute_improved_policy_value(
        self, values: torch.Tensor, idxs: Optional[torch.Tensor] = None
    ):
        grid_world = self.mdp_data.grid_world
        Pi = grid_world.improve_policy(values, idxs=idxs)
        *_, values = grid_world.evaluate_policy_iteratively(
            Pi=Pi, stop_at_rmse=self.stop_at_rmse, idxs=idxs
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
        q_values: torch.Tensor,
        values: torch.Tensor,
        **kwargs,
    ):
        _, max_n_bellman, _, _ = q_values.shape
        max_n_bellman -= 1
        v1 = values[:, 0]
        q1 = q_values[:, 0]
        final_outputs = torch.zeros_like(q1)
        all_outputs = []
        for j in range(iterations):
            outputs: torch.Tensor
            loss: torch.Tensor
            targets = q_values[:, min((j + 1) * bellman_delta, max_n_bellman)]
            with torch.no_grad():
                outputs, loss = net.forward(v1=v1, **kwargs, targets=targets)
            Pi = self._action_probs[idxs]
            v1 = (outputs * Pi).sum(-1)
            mask = (input_n_bellman + j * bellman_delta) < max_n_bellman
            final_outputs[mask] = outputs[mask]
            all_outputs.append(torch.stack([final_outputs, targets]))

        metrics = get_metrics(
            loss=loss,
            outputs=final_outputs,
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
                (idxs, input_n_bellman, action_probs, discrete, q_values, values) = [
                    x.cuda() for x in x
                ]
                metrics, outputs = self.get_n_metrics(
                    idxs=idxs,
                    input_n_bellman=input_n_bellman,
                    net=net,
                    q_values=q_values,
                    values=values,
                    action_probs=action_probs,
                    discrete=discrete,
                    **kwargs,
                )
                counter.update(metrics)
                all_outputs.append(outputs)
                all_idxs.append(idxs)
        metrics = {k: v / len(loader) for k, v in counter.items()}
        A = len(self.mdp_data.grid_world.deltas)

        # add values plots to metrics
        outputs = torch.cat(all_outputs, 2)
        idxs = torch.cat(all_idxs)
        values: torch.Tensor
        values = outputs[:, :, :, ::A]
        mask = torch.isin(idxs, plot_indices)
        idxs = idxs[mask].cpu()
        Pi = self.mdp_data.Pi[idxs][None, None].cuda()
        plot_values = values[:, :, mask]
        plot_values: torch.Tensor = plot_values * Pi
        plot_values = plot_values.sum(-1).cpu()
        plot_values = torch.unbind(plot_values, dim=2)
        for i, plot_value in zip(idxs, plot_values):
            fig = self.mdp_data.grid_world.visualize_values(plot_value)
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


def make(path: "str | Path", *args, **kwargs) -> Dataset:
    path = Path(path)
    name = path.stem
    name = ".".join(path.parts)
    module = importlib.import_module(name)
    data: Dataset = module.Dataset(*args, **kwargs)
    return data
