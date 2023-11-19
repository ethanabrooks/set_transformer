from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader

from dataset.base import Dataset as BaseDataset
from metrics import get_metrics
from models.value_unconditional import DataPoint, SetTransformer
from values.base import Values
from values.sample_uniform import Values as GroundTruthValues


@dataclass(frozen=True)
class Dataset(BaseDataset):
    max_n_bellman: int

    def __getitem__(self, idx) -> DataPoint:
        idx, n_bellman = np.unravel_index(idx, (len(self.sequence), self.max_n_bellman))
        transitions = self.sequence.transitions[idx]

        return DataPoint(
            action_probs=transitions.action_probs,
            actions=transitions.actions,
            idx=idx,
            n_bellman=n_bellman,
            next_states=transitions.next_states,
            q_values=self.Q[idx],
            rewards=transitions.rewards,
            states=transitions.states,
        )

    def __len__(self):
        return len(self.sequence) * self.max_n_bellman

    @classmethod
    def make(cls, values: Values, **kwargs):
        Q = values.Q
        return cls(**kwargs, max_n_bellman=len(Q), Q=Q.permute(1, 2, 0), values=values)

    def evaluate(
        self,
        bellman_number: int,
        ground_truth_values: GroundTruthValues,
        n_batch: int,
        net: SetTransformer,
        accuracy_threshold: float,
    ):
        net.eval()
        counter = Counter()
        loader = DataLoader(self, batch_size=n_batch, shuffle=False)

        with torch.no_grad():
            x: torch.Tensor
            for x in loader:
                x: DataPoint = DataPoint(*[x.cuda() for x in x])
                outputs: torch.Tensor
                with torch.no_grad():
                    outputs, _ = net.forward(x, q_values=x.q_values)

                def _get_metrics(
                    prefix: str, outputs: torch.Tensor, targets: torch.Tensor
                ):
                    metrics = get_metrics(
                        loss=None,
                        outputs=outputs,
                        targets=targets,
                        accuracy_threshold=accuracy_threshold,
                    )
                    return {f"{prefix}/{k}": v for k, v in asdict(metrics).items()}

                bootstrap_metrics = _get_metrics(
                    "(bootstrap)",
                    outputs=outputs,
                    targets=x.q_values,
                )
                targets = ground_truth_values.Q.cuda()[
                    bellman_number + 1, x.idx[:, None], x.states
                ]
                ground_truth_metrics = _get_metrics(
                    "(ground-truth)",
                    outputs=outputs,
                    targets=targets,
                )
                versus_metrics = _get_metrics(
                    "(botstrap versus ground-truth)",
                    outputs=x.q_values,
                    targets=targets,
                )
                # TODO: uncomment
                # if bellman_number == 0:
                assert torch.allclose(x.q_values, targets)
                for k, v in dict(
                    **bootstrap_metrics, **ground_truth_metrics, **versus_metrics
                ).items():
                    if v is not None:
                        counter.update({k: v})

        return {k: v / len(loader) for k, v in counter.items()}
