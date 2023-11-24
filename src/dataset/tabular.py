from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.utils.data
from torch.utils.data import DataLoader

import wandb
from dataset.base import Dataset as BaseDataset
from metrics import get_metrics
from models.tabular import DataPoint, SetTransformer


@dataclass(frozen=True)
class Dataset(BaseDataset):
    max_n_bellman: Optional[int]

    @property
    def n_bellman_convergance(self):
        return len(self.values.Q) - 1

    def evaluate(
        self, n_batch: int, net: SetTransformer, plot_indices: torch.Tensor, **kwargs
    ):
        net.eval()
        counter = Counter()
        loader = DataLoader(self, batch_size=n_batch, shuffle=False)
        grid_world = self.sequence.grid_world

        def generate():
            with torch.no_grad():
                x: torch.Tensor
                for x in loader:
                    x: DataPoint = DataPoint(*[x.cuda() for x in x])
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

    def get_max_n_bellman(self):
        if self.max_n_bellman is None:
            return self.n_bellman_convergance
        return self.max_n_bellman

    def get_metrics(
        self,
        accuracy_threshold: float,
        iterations: int,
        net: SetTransformer,
        x: DataPoint,
    ):
        assert torch.all(x.input_q == 0)

        def generate(input_q: torch.Tensor):
            for _ in range(iterations):
                with torch.no_grad():
                    input_q: torch.Tensor
                    input_q, _ = net.forward(x._replace(input_q=input_q, target_q=None))
                yield input_q

        outputs = torch.stack(list(generate(x.input_q)))
        idx = 1 + torch.arange(iterations)[:, None].cuda()
        idx = idx * self.bellman_delta + x.n_bellman[None]
        idx = torch.clamp(idx, 0, len(self.values.Q) - 1)
        targets = self.values.Q[idx.cpu(), x.idx[None].cpu()].cuda()

        metrics = get_metrics(
            loss=None,
            outputs=outputs[-1],
            targets=targets[-1],
            accuracy_threshold=accuracy_threshold,
        )
        metrics = asdict(metrics)
        return metrics, outputs, targets

    def input_q(self, idx: int, n_bellman: int):
        return self.values.Q[n_bellman, idx]

    def target_q(self, idx: int, n_bellman: int):
        n_bellman = n_bellman + self.bellman_delta
        n_bellman = min(n_bellman, len(self.values.Q) - 1)
        return self.values.Q[n_bellman, idx]
