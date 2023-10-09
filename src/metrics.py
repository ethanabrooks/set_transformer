from dataclasses import dataclass
from enum import Enum, auto

import torch
import torch.nn.functional as F


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
    outputs: torch.Tensor, targets: torch.Tensor, loss_type: LossType
) -> tuple[torch.Tensor, Metrics]:
    if loss_type == LossType.MSE:
        outputs = outputs.squeeze(-1)
        loss = F.mse_loss(outputs, targets.float())
        outputs = (100 * outputs).round() / 100
    elif loss_type == LossType.CROSS_ENTROPY:
        loss = F.cross_entropy(outputs.swapaxes(1, 2), targets)
        outputs = outputs.argmax(-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    mask = targets != 0
    accuracy = (outputs == targets)[mask].float().mean().item()
    within1accuracy = ((outputs - targets)[mask].abs() <= 1).float().mean().item()
    within2accuracy = ((outputs - targets)[mask].abs() <= 2).float().mean().item()
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
    return loss, metrics
