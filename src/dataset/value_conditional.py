from collections import Counter
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.utils.data
from torch.utils.data import DataLoader

import wandb
from dataset.base import Dataset as BaseDataset
from metrics import get_metrics
from models.value_conditional import DataPoint, SetTransformer
from sequence.base import Sequence
from values.base import Values


@dataclass(frozen=True)
class Dataset(BaseDataset):
    input_bellman: torch.Tensor
    max_n_bellman: int
    Q: torch.Tensor

    def __getitem__(self, idx) -> DataPoint:
        transitions = self.sequence.transitions[idx]

        obs = transitions.obs
        if obs is None:
            obs = transitions.states

        next_obs = transitions.next_obs
        if next_obs is None:
            next_obs = transitions.next_states

        return DataPoint(
            action_probs=transitions.action_probs,
            actions=transitions.actions,
            idx=idx,
            n_bellman=self.input_bellman[idx],
            next_obs=next_obs,
            next_states=transitions.next_states,
            obs=obs,
            q_values=self.Q[:, idx],
            rewards=transitions.rewards,
            states=transitions.states,
        )

    @classmethod
    def make(
        cls,
        max_initial_bellman: Optional[int],
        sequence: Sequence,
        values: Values,
    ):
        B = sequence.grid_world.n_tasks
        Q = values.Q

        # sample n_bellman -- number of steps of policy evaluation
        if max_initial_bellman is None:
            max_initial_bellman = len(Q) - 1
        input_bellman = torch.randint(0, max_initial_bellman, [B])

        return cls(
            input_bellman=input_bellman,
            max_n_bellman=len(Q) - 1,
            sequence=sequence,
            Q=Q,
            values=values,
        )

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

    def get_metrics(
        self,
        accuracy_threshold: float,
        bellman_delta: int,
        iterations: int,
        net: SetTransformer,
        x: DataPoint,
    ):
        input_q = self.index_values(x.q_values, x.n_bellman)

        assert torch.all(input_q == 0)

        def generate(input_q: torch.Tensor):
            for j in range(iterations):
                target_idxs = x.n_bellman + (j + 1) * bellman_delta
                q_values = self.index_values(x.q_values, target_idxs)
                with torch.no_grad():
                    input_q: torch.Tensor
                    input_q, _ = net.forward(x, input_q=input_q, target_q=q_values)
                yield input_q, q_values

        all_outputs, all_targets = zip(*generate(input_q))
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
