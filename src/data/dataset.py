from collections import Counter
from dataclasses import asdict, dataclass
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader

import wandb
from data.mdp import MDP
from metrics import get_metrics


class DataPoint(NamedTuple):
    idx: torch.Tensor
    input_bellman: torch.Tensor
    continuous: torch.Tensor
    discrete: torch.Tensor
    q_values: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class Dataset(torch.utils.data.Dataset):
    continuous: torch.Tensor
    discrete: torch.Tensor
    input_bellman: torch.Tensor
    max_n_bellman: int
    mdp: MDP
    omit_states_actions: int
    optimally_improved_policy_values: torch.Tensor
    Q: torch.Tensor
    stop_at_rmse: float
    V: torch.Tensor

    @classmethod
    def make(
        cls,
        max_initial_bellman: Optional[int],
        mdp: MDP,
        omit_states_actions: int,
        stop_at_rmse: float,
    ):
        transitions = mdp.transitions
        states = transitions.states
        action_probs = transitions.action_probs

        B = mdp.grid_world.n_tasks
        S = mdp.grid_world.n_states
        A = len(mdp.grid_world.deltas)
        Q = torch.stack(
            mdp.grid_world.evaluate_policy_iteratively(
                Pi=mdp.Pi, stop_at_rmse=stop_at_rmse
            )
        )

        # sample n_bellman -- number of steps of policy evaluation
        if max_initial_bellman is None:
            max_initial_bellman = len(Q) - 1
        input_bellman = torch.randint(0, max_initial_bellman, [B])
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

        V = (Q * mdp.Pi[None]).sum(-1)
        V_indexed = V[q_mesh, b_mesh]
        V_indexed = V_indexed[
            torch.arange(len(Q))[:, None, None],
            torch.arange(B)[None, :, None],
            states[None],
        ]

        continuous = torch.Tensor(action_probs)
        discrete = [
            transitions.states[..., None],
            transitions.actions[..., None],
            transitions.next_states[..., None],
            transitions.rewards[..., None],
        ]
        discrete = torch.cat(discrete, -1).long()

        permutation = torch.rand(B, S * A).argsort(dim=1)

        def permute(
            x: torch.Tensor,
            i: int,
            idxs: Optional[torch.Tensor] = None,
        ):
            p = permutation.to(x.device)
            if idxs is not None:
                p = p[idxs]
            for _ in range(i - 1):
                p = p[None]
            while p.dim() < x.dim():
                p = p[..., None]

            return torch.gather(x, i, p.expand_as(x))

        if omit_states_actions > 0:
            Q_indexed, V_indexed = [
                permute(x, 2)[:, :, omit_states_actions:]
                for x in [Q_indexed, V_indexed]
            ]
            continuous, discrete, action_probs = [
                permute(x, 1)[:, omit_states_actions:]
                for x in [continuous, discrete, action_probs]
            ]

        optimally_improved_policy_values = mdp.grid_world.evaluate_improved_policy(
            stop_at_rmse=stop_at_rmse,
            Q=Q[-1],
        ).cuda()
        return cls(
            continuous=continuous,
            discrete=discrete,
            input_bellman=input_bellman,
            max_n_bellman=len(Q) - 1,
            mdp=mdp,
            omit_states_actions=omit_states_actions,
            optimally_improved_policy_values=optimally_improved_policy_values,
            Q=Q_indexed,
            stop_at_rmse=stop_at_rmse,
            V=V_indexed,
        )

    @property
    def n_actions(self):
        return len(self.mdp.grid_world.deltas)

    def __getitem__(self, idx) -> DataPoint[int]:
        input_bellman = self.input_bellman[idx]
        return DataPoint(
            idx=idx,
            input_bellman=input_bellman,
            continuous=self.continuous[idx],
            discrete=self.discrete[idx],
            q_values=self.Q[:, idx],
            values=self.V[:, idx],
        )

    def __len__(self):
        return len(self.discrete)

    def compute_improved_policy_value(
        self, values: torch.Tensor, idxs: Optional[torch.Tensor] = None
    ):
        return self.mdp.grid_world.evaluate_improved_policy(
            stop_at_rmse=self.stop_at_rmse,
            Q=values,
            idxs=idxs,
        )

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
                x: DataPoint[torch.Tensor] = DataPoint(*[x.cuda() for x in x])
                metrics, outputs = self.get_metrics(
                    idxs=x.idx,
                    input_n_bellman=x.input_bellman,
                    net=net,
                    q_values=x.q_values,
                    values=x.values,
                    continuous=x.continuous,
                    discrete=x.discrete,
                    **kwargs,
                )
                counter.update(metrics)
                all_outputs.append(outputs)
                all_idxs.append(x.idx)
        metrics = {k: v / len(loader) for k, v in counter.items()}
        A = len(self.mdp.grid_world.deltas)

        # add values plots to metrics
        outputs = torch.cat(all_outputs, 2)
        idxs = torch.cat(all_idxs)
        values: torch.Tensor
        values = outputs[:, :, :, ::A]
        mask = torch.isin(idxs, plot_indices)
        idxs = idxs[mask].cpu()
        Pi = self.mdp.Pi[idxs][None, None].cuda()
        plot_values = values[:, :, mask]
        plot_values: torch.Tensor = plot_values * Pi
        plot_values = plot_values.sum(-1).cpu()
        plot_values = torch.unbind(plot_values, dim=2)
        for i, plot_value in zip(idxs, plot_values):
            fig = self.mdp.grid_world.visualize_values(plot_value)
            metrics[f"values-plot {i}"] = wandb.Image(fig)

        return metrics

    def get_metrics(
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
        assert torch.all(v1 == 0)
        q1 = q_values[:, 0]
        assert torch.all(q1 == 0)
        final_outputs = torch.zeros_like(q1)
        all_outputs = []
        Pi = self.mdp.transitions.action_probs.cuda()[idxs]
        for j in range(iterations):
            outputs: torch.Tensor
            loss: torch.Tensor
            targets = q_values[:, min((j + 1) * bellman_delta, max_n_bellman)]
            with torch.no_grad():
                outputs, loss = net.forward(v1=v1, **kwargs, targets=targets)
            v1: torch.Tensor = outputs * Pi
            v1 = v1.sum(-1)
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
