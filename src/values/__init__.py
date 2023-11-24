from pathlib import Path

import torch

from artifacts import download_and_return_root, get_run
from metrics import compute_rmse
from sequence.base import Sequence
from values.cross_product import Values as SampleUniformValues
from values.tabular import Values


def make(
    load_path: str, sample_from_trajectories: bool, sequence: Sequence, **kwargs
) -> Values:
    if load_path is None:
        Q = None
    else:
        type = "Q"
        run = get_run(load_path)
        artifact_root = download_and_return_root(load_path=load_path, type=type)
        artifact_path: Path = artifact_root / f"{type}.pt"
        Q = torch.load(artifact_path)
        ground_truth = torch.gather(
            sequence.grid_world.Q,
            dim=2,
            index=sequence.transitions.states[None, ..., None].expand_as(Q),
        )
        rmse = compute_rmse(Q, ground_truth)
        assert rmse <= run.config["rmse_training_final"], "RMSE too high"
    kwargs.update(Q=Q, sequence=sequence)

    return (
        Values.make(**kwargs)
        if sample_from_trajectories
        else SampleUniformValues.make(**kwargs)
    )
