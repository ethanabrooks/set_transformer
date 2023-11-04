from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

import wandb
from metrics import get_metrics
from models.set_transformer import DataPoint
from sequence.base import Sequence
from values.base import Values


@dataclass(frozen=True)
class Dataset(BaseDataset):
    continuous: torch.Tensor
    discrete: torch.Tensor
    input_bellman: torch.Tensor
    max_n_bellman: int
    sequence: Sequence
    Q: torch.Tensor
    V: torch.Tensor
    values: Values

    @classmethod
    def make(
        cls,
        max_initial_bellman: Optional[int],
        sequence: Sequence,
        values: Values,
    ):
        transitions = sequence.transitions
        B = sequence.grid_world.n_tasks
        Q = values.Q
        V = (Q * transitions.action_probs[None]).sum(-1)

        # sample n_bellman -- number of steps of policy evaluation
        if max_initial_bellman is None:
            max_initial_bellman = len(Q) - 1
        input_bellman = torch.randint(0, max_initial_bellman, [B])
        n_bellman = torch.arange(len(Q))[:, None] + input_bellman[None, :]
        n_bellman = torch.clamp(n_bellman, 0, len(Q) - 1)

        continuous = torch.Tensor(transitions.action_probs)
        discrete = [
            transitions.states[..., None],
            transitions.actions[..., None],
            transitions.next_states[..., None],
            transitions.rewards[..., None],
        ]
        discrete = torch.cat(discrete, -1).long()

        return cls(
            continuous=continuous,
            discrete=discrete,
            input_bellman=input_bellman,
            max_n_bellman=len(Q) - 1,
            sequence=sequence,
            Q=Q,
            V=V,
            values=values,
        )

    @property
    def n_actions(self):
        return len(self.sequence.grid_world.deltas)

    def __getitem__(self, idx) -> DataPoint[int]:
        return DataPoint(
            idx=idx,
            input_bellman=self.input_bellman[idx],
            continuous=self.continuous[idx],
            discrete=self.discrete[idx],
            q_values=self.Q[:, idx],
            values=self.V[:, idx],
        )

    def __len__(self):
        return len(self.discrete)

    def evaluate(
        self, n_batch: int, net: nn.Module, plot_indices: torch.Tensor, **kwargs
    ):
        net.eval()
        counter = Counter()
        loader = DataLoader(self, batch_size=n_batch, shuffle=False)
        grid_world = self.sequence.grid_world

        def generate():
            with torch.no_grad():
                x: torch.Tensor
                for x in loader:
                    x: DataPoint[torch.Tensor] = DataPoint(*[x.cuda() for x in x])
                    metrics, outputs, targets = self.get_metrics(net=net, **kwargs, x=x)
                    metrics.update(self.values.get_metrics(idxs=x.idx, outputs=outputs))
                    for k, v in metrics.items():
                        if v is not None:
                            counter.update({k: v})
                    yield x.idx, outputs, targets

        A = len(grid_world.deltas)

        # add values plots to metrics
        all_idxs, all_outputs, all_targets = zip(*generate())
        idxs = torch.cat(all_idxs)
        outputs = torch.cat(all_outputs, 1)
        targets = torch.cat(all_targets, 1)
        stacked = torch.stack([outputs, targets], 1)
        values = stacked[..., ::A, :]
        mask = torch.isin(idxs, plot_indices)
        idxs = idxs[mask].cpu()
        Pi = grid_world.Pi[idxs][None, None].cuda()
        plot_values = values[:, :, mask]
        plot_values: torch.Tensor = plot_values * Pi
        plot_values = plot_values.sum(-1).cpu()
        plot_values = torch.unbind(plot_values, dim=2)
        metrics = {k: v / len(loader) for k, v in counter.items()}
        for i, plot_value in zip(idxs, plot_values):
            fig = grid_world.visualize_values(plot_value)
            metrics[f"values-plot {i}"] = wandb.Image(fig)

        return metrics

    def get_metrics(
        self,
        accuracy_threshold: float,
        bellman_delta: int,
        iterations: int,
        net: nn.Module,
        x: DataPoint[torch.Tensor],
    ):
        v1 = self.index_values(x.values, x.input_bellman)

        assert torch.all(v1 == 0)
        Pi = self.sequence.transitions.action_probs.cuda()[x.idx]

        def generate(v1: torch.Tensor):
            for j in range(iterations):
                outputs: torch.Tensor
                target_idxs = x.input_bellman + (j + 1) * bellman_delta
                targets = self.index_values(x.q_values, target_idxs)
                with torch.no_grad():
                    outputs, _ = net.forward(
                        continuous=x.continuous, discrete=x.discrete, v1=v1
                    )
                v1: torch.Tensor = outputs * Pi
                v1 = v1.sum(-1)
                yield outputs, targets

        all_outputs, all_targets = zip(*generate(v1))
        outputs = torch.stack(all_outputs)
        targets = torch.stack(all_targets)

        metrics = get_metrics(
            loss=None,
            outputs=outputs[-1],
            targets=targets[-1],
            accuracy_threshold=accuracy_threshold,
        )
        metrics = asdict(metrics)
        return metrics, outputs, targets

    def index_values(self, values: torch.Tensor, idxs: torch.Tensor):
        idxs = torch.clamp(idxs, 0, self.max_n_bellman)
        return values[torch.arange(len(values)), idxs]
