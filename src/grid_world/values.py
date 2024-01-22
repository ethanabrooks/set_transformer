from copy import deepcopy
from dataclasses import asdict, dataclass, replace

import torch
from tqdm import tqdm

from grid_world.base import GridWorld
from utils import tensor_hash


@dataclass(frozen=True)
class GridWorldWithValues(GridWorld):
    optimally_improved_policy_values: torch.Tensor
    Q: torch.Tensor
    stop_at_rmse: float

    @classmethod
    def make(cls, stop_at_rmse: float, grid_world: GridWorld, verbose: bool = False):
        iterator = grid_world.evaluate_policy_iteratively(
            Pi=grid_world.Pi, stop_at_rmse=stop_at_rmse
        )
        if verbose:
            iterator = tqdm(iterator, desc="Computing values")
        Q = torch.stack(list(iterator))
        optimally_improved_policy_values = grid_world.evaluate_improved_policy(
            Q=Q[-1], stop_at_rmse=stop_at_rmse
        )
        return cls(
            **asdict(replace(grid_world)),
            optimally_improved_policy_values=optimally_improved_policy_values,
            Q=Q,
            stop_at_rmse=stop_at_rmse,
        )

    def __getitem__(self, idx: torch.Tensor):
        self = deepcopy(self)

        def to_device(x: torch.Tensor):
            return x.to(idx.device) if isinstance(idx, torch.Tensor) else x

        item = GridWorld.__getitem__(self, idx)
        optimally_improved_policy_values = to_device(
            self.optimally_improved_policy_values
        )[idx]
        Q = to_device(self.Q)[:, idx]

        return type(self)(
            **asdict(item),
            optimally_improved_policy_values=optimally_improved_policy_values,
            Q=Q,
            stop_at_rmse=self.stop_at_rmse,
        )

    def __hash__(self):
        hashcode = super().__hash__()
        return hash(
            (
                hashcode,
                tensor_hash(self.optimally_improved_policy_values),
                tensor_hash(self.Q),
                self.stop_at_rmse,
            )
        )
