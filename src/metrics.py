from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Metrics:
    loss: float
    mae: float
    argmax_accuracy: float
    rmse: float


def compute_rmse(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (outputs - targets).square().float().mean(-1).sqrt().mean().item()


def compute_mae(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.abs(outputs - targets).float().mean().item()


def get_metrics(
    loss: Optional[torch.Tensor],
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> Metrics:
    mae = compute_mae(outputs, targets)
    rmse = compute_rmse(outputs, targets)
    argmax_accuracy = (outputs.argmax(-1) == targets.argmax(-1)).float().mean().item()

    metrics = Metrics(
        loss=None if loss is None else loss.item(),
        mae=mae,
        argmax_accuracy=argmax_accuracy,
        rmse=rmse,
    )
    return metrics
