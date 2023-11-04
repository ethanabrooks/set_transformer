from dataclasses import dataclass

import torch
from tqdm import tqdm

from tabular.grid_world import GridWorld


@dataclass
class GridWorldWithValues(GridWorld):
    optimally_improved_policy_values: torch.Tensor
    Q: torch.Tensor
    stop_at_rmse: float

    @classmethod
    def make(
        cls,
        absorbing_state: bool,
        dense_reward: bool,
        gamma: float,
        grid_size: int,
        heldout_goals: list[tuple[int, int]],
        n_maze: int,
        n_tasks: int,
        p_wall: float,
        seed: int,
        stop_at_rmse: float,
        terminate_on_goal: bool,
        use_heldout_goals: bool,
    ):
        grid_world = GridWorld.make(
            absorbing_state=absorbing_state,
            dense_reward=dense_reward,
            gamma=gamma,
            grid_size=grid_size,
            heldout_goals=heldout_goals,
            n_maze=n_maze,
            n_tasks=n_tasks,
            p_wall=p_wall,
            seed=seed,
            terminate_on_goal=terminate_on_goal,
            use_heldout_goals=use_heldout_goals,
        )
        Q = torch.stack(
            list(
                tqdm(
                    grid_world.evaluate_policy_iteratively(
                        Pi=grid_world.Pi, stop_at_rmse=stop_at_rmse
                    ),
                    desc="Computing values",
                )
            )
        )
        optimally_improved_policy_values = grid_world.evaluate_improved_policy(
            Q=Q[-1], stop_at_rmse=stop_at_rmse
        ).cuda()
        return cls(
            deltas=grid_world.deltas,
            dense_reward=grid_world.dense_reward,
            gamma=grid_world.gamma,
            goals=grid_world.goals,
            grid_size=grid_world.grid_size,
            heldout_goals=grid_world.heldout_goals,
            is_wall=grid_world.is_wall,
            n_tasks=grid_world.n_tasks,
            Pi=grid_world.Pi,
            random=grid_world.random,
            states=grid_world.states,
            terminate_on_goal=grid_world.terminate_on_goal,
            use_absorbing_state=grid_world.absorbing_state,
            use_heldout_goals=grid_world.use_heldout_goals,
            optimally_improved_policy_values=optimally_improved_policy_values,
            Q=Q,
            stop_at_rmse=stop_at_rmse,
        )

    def __getitem__(self, idx: torch.Tensor):
        def to_device(x: torch.Tensor):
            return x.to(idx.device) if isinstance(idx, torch.Tensor) else x

        item = GridWorld.__getitem__(self, idx)

        return type(self)(
            deltas=item.deltas,
            dense_reward=item.dense_reward,
            gamma=item.gamma,
            goals=item.goals,
            grid_size=item.grid_size,
            heldout_goals=item.heldout_goals,
            is_wall=item.is_wall,
            n_tasks=item.n_tasks,
            Pi=item.Pi,
            random=item.random,
            states=item.states,
            terminate_on_goal=item.terminate_on_goal,
            use_absorbing_state=item.use_absorbing_state,
            use_heldout_goals=item.use_heldout_goals,
            optimally_improved_policy_values=to_device(
                self.optimally_improved_policy_values
            )[idx],
            Q=to_device(self.Q)[:, idx],
            stop_at_rmse=self.stop_at_rmse,
        )
