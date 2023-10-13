from dataclasses import dataclass
from enum import Enum, auto

import torch


@dataclass
class Metrics:
    loss: float
    mae: float
    pair_wise_accuracy: float
    rmse: float


class LossType(Enum):
    MSE = auto()
    CROSS_ENTROPY = auto()


def compute_rmse(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return (outputs - targets).square().float().mean(-1).sqrt().mean().item()


def compute_mae(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.abs(outputs - targets).float().mean().item()


def get_metrics(
    decode_outputs: torch.Tensor,
    loss: torch.Tensor,
    loss_type: LossType,
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> Metrics:
    if loss_type == LossType.MSE:
        outputs = outputs.squeeze(-1)
    elif loss_type == LossType.CROSS_ENTROPY:
        outputs = outputs.argmax(-1)
        decode_outputs = decode_outputs.cuda()
        outputs = decode_outputs[outputs]
        targets = decode_outputs[targets]
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    mae = compute_mae(outputs, targets)
    rmse = compute_rmse(outputs, targets)

    outputs = (50 * outputs).round() / 50

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
        mae=mae,
        pair_wise_accuracy=pair_wise_accuracy,
        rmse=rmse,
    )
    return metrics
