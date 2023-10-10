from dataclasses import dataclass
from enum import Enum, auto

import torch


@dataclass
class Metrics:
    accuracy: float
    loss: float
    pair_wise_accuracy: float
    within1accuracy: float
    within2accuracy: float


class LossType(Enum):
    MSE = auto()
    CROSS_ENTROPY = auto()


def get_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
    loss_type: LossType,
) -> Metrics:
    if loss_type == LossType.MSE:
        outputs = outputs.squeeze(-1)
        outputs = (100 * outputs).round() / 100
        unit = 0.01
    elif loss_type == LossType.CROSS_ENTROPY:
        outputs = outputs.argmax(-1)
        unit = 1
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    mask = targets != 0

    def process_accuracy(accuracy: torch.Tensor) -> torch.Tensor:
        return accuracy[mask].float().mean().item()

    accuracy = process_accuracy(outputs == targets)
    within1accuracy = process_accuracy((outputs - targets).abs() <= 1 * unit)
    within2accuracy = process_accuracy((outputs - targets).abs() <= 2 * unit)

    # Compute pairwise differences for outputs and targets
    diff_outputs = outputs[:, 1:] - outputs[:, :-1]
    diff_targets = targets[:, 1:] - targets[:, :-1]

    # Compute signs of differences
    sign_outputs = torch.sign(diff_outputs)
    sign_targets = torch.sign(diff_targets)

    # Count where signs match
    pair_wise_accuracy = (sign_outputs == sign_targets).float().mean().item()
    metrics = Metrics(
        loss=loss.item(),
        accuracy=accuracy,
        pair_wise_accuracy=pair_wise_accuracy,
        within1accuracy=within1accuracy,
        within2accuracy=within2accuracy,
    )
    return metrics
