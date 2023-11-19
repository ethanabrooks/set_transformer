import itertools
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from metrics import compute_rmse
from tabular.maze import generate_maze, maze_to_state_action
from utils import Transition, tensor_hash


def convert_2d_to_1d(grid_size: int, x: torch.Tensor):
    return grid_size * x[..., 0] + x[..., 1]


def convert_1d_to_2d(grid_size: int, x: torch.Tensor):
    return torch.stack([x // grid_size, x % grid_size], dim=1)


def evaluate_policy(
    gamma: float,
    Pi: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    T: torch.Tensor,
    D: torch.Tensor,
):
    B, N, A = Pi.shape

    assert [*R.shape] == [B, N, A]
    EQ = (Pi * Q).sum(-1)
    assert [*EQ.shape] == [B, N]
    EQ = (T * EQ[:, None, None]).sum(-1)
    assert [*EQ.shape] == [B, N, A]
    Q = R + gamma * ~D * EQ
    return Q


@dataclass(frozen=True)
class GridWorld:
    deltas: torch.Tensor
    dense_reward: bool
    gamma: float
    goals: torch.Tensor
    grid_size: int
    hashcode: int
    heldout_goals: list[tuple[int, int]]
    is_wall: torch.Tensor
    n_tasks: int
    Pi: torch.Tensor
    random: np.random.Generator
    states: torch.Tensor
    terminal_transitions: Optional[torch.Tensor]
    terminate_on_goal: bool
    use_absorbing_state: bool
    use_heldout_goals: bool

    @classmethod
    def compute_hashcode(
        cls,
        deltas: torch.Tensor,
        gamma: float,
        goals: torch.Tensor,
        grid_size: int,
        is_wall: torch.Tensor,
        n_tasks: int,
        Pi: torch.Tensor,
        random: np.random.Generator,
        states: torch.Tensor,
        terminal_transitions: Optional[torch.Tensor],
        terminate_on_goal: bool,
        use_heldout_goals: bool,
    ):
        return hash(
            (
                tensor_hash(deltas),
                gamma,
                tensor_hash(goals),
                grid_size,
                tensor_hash(is_wall),
                n_tasks,
                tensor_hash(Pi),
                random,
                tensor_hash(states),
                None
                if terminal_transitions is None
                else tensor_hash(terminal_transitions),
                terminate_on_goal,
                use_heldout_goals,
            )
        )

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
        terminal_transitions: Optional[torch.Tensor],
        terminate_on_goal: bool,
        use_heldout_goals: bool,
    ):
        # transition to absorbing state instead of goal
        deltas = torch.tensor([[0, 1], [0, -1], [-1, 0], [1, 0]])
        A = len(deltas)
        G = grid_size**2  # number of goals
        S = G
        if absorbing_state:
            S += 1
        M = n_maze
        B = n_tasks

        # add absorbing state for goals
        random = np.random.default_rng(seed)

        states = torch.tensor(
            [[i, j] for i in range(grid_size) for j in range(grid_size)]
        )

        # generate walls
        is_wall = torch.rand(B, G, A) < p_wall
        if n_maze:
            mazes = [
                maze_to_state_action(generate_maze(grid_size)).view(G, A)
                for _ in tqdm(range(M), desc="Generating mazes")
            ]
            mazes = torch.stack(mazes)
            assert [*mazes.shape] == [M, G, A]
            maze_idx = torch.randint(0, M, (B,))
            is_wall = mazes[maze_idx] & is_wall
            assert [*is_wall.shape] == [B, G, A]
        goals = torch.randint(0, G, (B,))

        alpha = torch.ones(A)
        Pi: torch.Tensor = torch.distributions.Dirichlet(alpha).sample(
            (B, S)
        )  # random policies
        assert [*Pi.shape] == [B, S, A]
        hashcode = cls.compute_hashcode(
            deltas=deltas,
            gamma=gamma,
            goals=goals,
            grid_size=grid_size,
            is_wall=is_wall,
            n_tasks=n_tasks,
            Pi=Pi,
            random=random,
            states=states,
            terminal_transitions=terminal_transitions,
            terminate_on_goal=terminate_on_goal,
            use_heldout_goals=use_heldout_goals,
        )

        return cls(
            deltas=deltas,
            dense_reward=dense_reward,
            gamma=gamma,
            goals=goals,
            grid_size=grid_size,
            hashcode=hashcode,
            heldout_goals=heldout_goals,
            is_wall=is_wall,
            n_tasks=n_tasks,
            Pi=Pi,
            random=random,
            states=states,
            terminal_transitions=terminal_transitions,
            terminate_on_goal=terminate_on_goal,
            use_absorbing_state=absorbing_state,
            use_heldout_goals=use_heldout_goals,
        )

    @property
    def absorbing_state(self):
        return self.n_states - 1

    @property
    def n_actions(self):
        return len(self.deltas)

    @property
    def n_states(self):
        n_states = self.grid_size**2
        if self.use_absorbing_state:
            n_states += 1
        return n_states

    @property
    def next_state_no_absorbing(self):
        A = self.n_actions
        G = self.n_states
        if self.use_absorbing_state:
            G -= 1
        T = self.n_tasks
        # Compute next states for each action and state for each batch (goal)
        next_states = self.states[:, None] + self.deltas[None, :]
        assert [*next_states.shape] == [G, A, 2]
        states = self.states[None, :, None].tile(T, 1, A, 1)
        next_states = next_states[None].tile(T, 1, 1, 1)
        is_wall = self.is_wall[..., None]
        next_states = states * is_wall + next_states * (~is_wall)
        next_states = torch.clamp(next_states, 0, self.grid_size - 1)  # stay in bounds
        return self.convert_2d_to_1d(next_states)

    @property
    def next_states(self):
        G = self.n_states - 1
        next_state = self.next_state_no_absorbing.clone()
        is_goal: torch.Tensor = self.goals[:, None, None] == next_state
        if self.use_absorbing_state:
            # S_[is_goal[..., None].expand_as(S_)] = absorbing_state_idx
            next_state[is_goal[:, :G]] = self.absorbing_state

            # Insert row for absorbing state
            padding = (0, 0, 0, 1)  # left 0, right 0, top 0, bottom 1
            next_state = F.pad(next_state, padding, value=self.absorbing_state)
        return next_state

    @property
    def is_goal(self):
        return self.goals[:, None, None] == self.next_state_no_absorbing

    @property
    @lru_cache()
    def rewards(self):
        goals = self.goals
        if self.dense_reward:
            distance = (goals[:, None] - self.states[None]).abs().sum(-1)
            R = -distance.float()[..., None].tile(1, 1, self.n_actions)
        else:
            R = self.is_goal.float()
        R = R
        if self.use_absorbing_state:
            padding = (0, 0, 0, 1)  # left 0, right 0, top 0, bottom 1
            R = F.pad(R, padding, value=0)  # Insert row for absorbing state
        return R

    @property
    @lru_cache()
    def termination_matrix(self):
        matrix = (
            self.rewards.bool()
            if self.terminate_on_goal
            else torch.zeros_like(self.rewards, dtype=torch.bool)
        )
        if self.terminal_transitions is not None:
            matrix = matrix | self.terminal_transitions
        return matrix

    @property
    @lru_cache()
    def transition_matrix(self):
        matrix: torch.Tensor = F.one_hot(self.next_states, num_classes=self.n_states)
        return matrix.float()

    def __hash__(self):
        return self.hashcode

    def __getitem__(self, idx: torch.Tensor):
        self = deepcopy(self)

        def to_device(x: torch.Tensor):
            return x.to(idx.device) if isinstance(idx, torch.Tensor) else x

        deltas = to_device(self.deltas)
        goals = to_device(self.goals)[idx]
        is_wall = to_device(self.is_wall)[idx]
        n_tasks = goals.numel()
        Pi = to_device(self.Pi)[idx]
        states = to_device(self.states)
        terminal_transitions = (
            None
            if self.terminal_transitions is None
            else to_device(self.terminal_transitions)[idx]
        )
        hashcode = self.compute_hashcode(
            deltas=deltas,
            gamma=self.gamma,
            goals=goals,
            grid_size=self.grid_size,
            is_wall=is_wall,
            n_tasks=n_tasks,
            Pi=Pi,
            random=self.random,
            states=states,
            terminal_transitions=terminal_transitions,
            terminate_on_goal=self.terminate_on_goal,
            use_heldout_goals=self.use_heldout_goals,
        )
        return GridWorld(
            deltas=to_device(self.deltas),
            dense_reward=self.dense_reward,
            gamma=self.gamma,
            goals=goals,
            grid_size=self.grid_size,
            hashcode=hashcode,
            heldout_goals=self.heldout_goals,
            is_wall=is_wall,
            n_tasks=n_tasks,
            Pi=Pi,
            random=self.random,
            states=states,
            terminal_transitions=terminal_transitions,
            terminate_on_goal=self.terminate_on_goal,
            use_absorbing_state=self.use_absorbing_state,
            use_heldout_goals=self.use_heldout_goals,
        )

    def __len__(self):
        return self.n_tasks

    def arange(self, shape: torch.Size):
        arange = torch.arange(self.n_tasks)
        while arange.dim() < len(shape):
            arange = arange[..., None]
        return arange.expand(shape).flatten()

    def convert_1d_to_2d(self, x: torch.Tensor):
        return convert_1d_to_2d(self.grid_size, x)

    def convert_2d_to_1d(self, x: torch.Tensor):
        return convert_2d_to_1d(self.grid_size, x)

    def create_exploration_policy(self):
        N = self.grid_size
        A = self.n_actions

        def odd(n):
            return bool(n % 2)

        assert not odd(N), "Perfect exploration only possible with even grid."

        # Initialize the policy tensor with zeros
        policy_2d = torch.zeros(N, N, A)

        # Define the deterministic policy
        for i in range(N):
            top = i == 0
            bottom = i == N - 1
            if top:
                up = None
            else:
                up = 0

            for j in range(N):
                if odd(i):
                    down = 1
                    move = 2  # left
                else:  # even i
                    down = N - 1
                    move = 3  # right

                if bottom:
                    down = None

                if j == up:
                    policy_2d[i, j, 0] = 1  # move up
                elif j == down:
                    policy_2d[i, j, 1] = 1  # move down
                else:
                    policy_2d[i, j, move] = 1  # move left/right

        # Flatten the 2D policy tensor to 1D
        policy = policy_2d.view(N * N, A)
        if self.use_absorbing_state:
            # Insert row for absorbing state
            policy = F.pad(policy, (0, 0, 0, 1), value=0)
            policy[-1, 0] = 1  # last state is terminal
        # self.visualize_policy(policy[None].tile(self.n_tasks, 1, 1))
        return policy

    def evaluate_improved_policy(
        self,
        stop_at_rmse: float,
        Q: torch.Tensor,
    ) -> torch.Tensor:
        Pi = torch.zeros_like(Q)
        Pi.scatter_(-1, Q.argmax(dim=-1, keepdim=True), 1.0)
        Q: torch.Tensor
        *_, Q = self.evaluate_policy_iteratively(Pi=Pi, stop_at_rmse=stop_at_rmse)
        return Q

    def evaluate_policy(
        self,
        Pi: torch.Tensor,
        Q: torch.Tensor = None,
    ):
        B = self.n_tasks
        N = self.n_states
        A = self.n_actions
        if Q is None:
            Q = torch.zeros((B, N, A), dtype=torch.float32, device=Pi.device)
        return evaluate_policy(
            gamma=self.gamma,
            Pi=Pi,
            Q=Q,
            R=self.rewards.to(Pi.device),
            T=self.transition_matrix.to(Pi.device),
            D=self.termination_matrix.to(Pi.device),
        )

    def evaluate_policy_iteratively(
        self,
        Pi: torch.Tensor,
        stop_at_rmse: float,
    ):
        B = self.n_tasks
        S = self.n_states
        A = self.n_actions

        Q = torch.zeros((B, S, A), device=Pi.device, dtype=torch.float)
        rmse = float("inf")
        for _ in itertools.count(1):  # n_rounds of policy evaluation
            yield Q
            if rmse < stop_at_rmse:
                break
            Q1 = self.evaluate_policy(Pi, Q)
            rmse = compute_rmse(Q1, Q)
            # print("Iteration:", k, "RMSE:", rmse)
            Q = Q1

    def get_trajectories(
        self,
        episode_length: int,
        n_episodes: int,
        use_exploration_policy: bool = False,
    ):
        B = self.n_tasks
        N = self.n_states
        A = self.n_actions
        assert [*self.Pi.shape] == [B, N, A]

        trajectory_length = episode_length * n_episodes
        states = torch.zeros((B, trajectory_length, 2), dtype=torch.int)
        actions = torch.zeros((B, trajectory_length), dtype=torch.int)
        action_probs = torch.zeros((B, trajectory_length, A), dtype=torch.int)
        next_states = torch.zeros((B, trajectory_length, 2), dtype=torch.int)
        rewards = torch.zeros((B, trajectory_length))
        done = torch.zeros((B, trajectory_length), dtype=torch.bool)
        S1 = self.reset_fn()
        arange = torch.arange(B)
        if use_exploration_policy:
            assert self.grid_size == 2, "Exploration policy only works for 2x2 grid"
            action_sequence = torch.tensor(
                [0, 3, 1, 2, 2, 1, 3, 1, 3, 0, 3, 0, 2, 0, 2, 1]
            )
            current_state_indices = self.convert_2d_to_1d(S1)
            start_index = torch.tensor([0, 1, 3, 2])[current_state_indices]

        for t in tqdm(range(trajectory_length), desc="Sampling trajectories"):
            # Convert current current_states to indices
            current_state_indices = self.convert_2d_to_1d(S1)

            # Sample actions from the policy
            if use_exploration_policy:
                A = action_sequence[(t + start_index) % len(action_sequence)]
            else:
                A = (
                    torch.multinomial(self.Pi[arange, current_state_indices], 1)
                    .squeeze(1)
                    .long()
                )

            # Convert current current_states to indices
            next_state_indices, R, D, _ = self.step_fn(
                states=current_state_indices, actions=A
            )

            # Convert next state indices to coordinates
            S2 = self.convert_1d_to_2d(next_state_indices)

            # Store the current current_states and rewards
            states[:, t] = S1
            actions[:, t] = A
            action_probs[:, t] = self.Pi[arange, current_state_indices]
            next_states[:, t] = S2
            rewards[:, t] = R
            done[:, t] = D

            # Update current current_states
            S1 = S2
            if not use_exploration_policy:
                S_reset = self.reset_fn()
                S1[D] = S_reset[D]

        return Transition(
            states=states,
            actions=actions[..., None],
            action_probs=action_probs,
            next_states=next_states,
            rewards=rewards,
            done=done,
        )

    def reset_fn(self):
        array = self.random.choice(self.grid_size, size=(self.n_tasks, 2))
        return torch.tensor(array)

    def step_fn(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ):
        # self.check_states(states)
        # self.check_actions(actions)
        assert len(states) == self.n_tasks
        assert states.shape == actions.shape
        shape = states.shape

        arange = self.arange(shape)
        states = states.flatten()
        actions = actions.flatten()

        rewards = self.rewards[arange, states, actions]

        # Compute next state indices
        next_states = torch.argmax(
            self.transition_matrix[arange, states, actions], dim=1
        )

        done = self.termination_matrix[arange, states, actions]
        next_states = next_states.reshape(shape)
        rewards = rewards.reshape(shape)
        done = done.reshape(shape)
        return next_states, rewards, done, {}

    def visualize_policy(self, Pi: torch.Tensor):
        dims = len(Pi.shape)

        if dims == 2:
            fig, ax = plt.subplots(figsize=(6, 6))
            plot_policy(Pi, ax, self.grid_size, self.deltas)

        elif dims == 3:
            n_tasks = Pi.shape[0]
            fig, axes = plt.subplots(1, n_tasks, figsize=(6 * n_tasks, 6))
            for idx, ax in enumerate(axes):
                plot_policy(Pi[idx], ax, self.grid_size, self.deltas)

        elif dims == 4:
            n_rows, n_cols = Pi.shape[:2]
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
            for i in range(n_rows):
                for j in range(n_cols):
                    plot_policy(Pi[i, j], axes[i, j], self.grid_size, self.deltas)
        else:
            raise ValueError(f"Unsupported number of dimensions: {dims}")

        plt.tight_layout()
        plt.savefig("policy.png")

    def visualize_values(self, V: torch.Tensor, save_path: Optional[str] = None):
        dims = len(V.shape)
        global_min = V.min().item()
        global_max = V.max().item()

        def imshow(values: torch.Tensor, ax: plt.Axes):
            n, _ = values.shape
            if self.use_absorbing_state:
                values = values[..., :-1]
            im = ax.imshow(
                values.reshape((n * self.grid_size, self.grid_size)),
                cmap="hot",
                interpolation="nearest",
                vmin=global_min,
                vmax=global_max,
            )
            ax.axis("off")  # Turn off the axes
            return im

        if dims == 2:
            fig, ax = plt.subplots(figsize=(self.grid_size, self.grid_size))
            imshow(V, ax)

        elif dims == 3:
            n_tasks = V.shape[0]
            fig, axes = plt.subplots(
                1, n_tasks, figsize=(self.grid_size * n_tasks, self.grid_size)
            )
            if n_tasks == 1:
                axes = [axes]
            for idx, ax in enumerate(axes):
                imshow(V[idx], ax)

        elif dims == 4:
            n_rows, n_cols = V.shape[:2]
            # Adjust the subplot size depending on the number of rows
            fig_size = (self.grid_size / (n_rows**0.5)) * n_cols, (
                self.grid_size / (n_rows**0.5)
            ) * n_rows
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=fig_size, sharex="all", sharey="all"
            )

            # Capture the return image for colorbar
            ims = []
            for i in range(n_rows):
                for j in range(n_cols):
                    im = imshow(V[i, j], axes[i, j])
                    ims.append(im)

            # Add a single colorbar for all plots
            # fig.colorbar(ims[0], ax=axes.ravel().tolist())

        else:
            raise ValueError(f"Unsupported number of dimensions: {dims}")

        if save_path:
            plt.savefig(
                save_path, dpi=self.grid_size**2 / 2
            )  # adjust dpi if necessary
        return fig


def plot_policy(policy: torch.Tensor, ax: plt.Axes, grid_size: int, deltas: list):
    N = grid_size
    plt.xlim(0, N)  # Adjusted the starting limit
    plt.ylim(0, N)

    # Draw grid
    for i in range(N + 1):
        offset = 0.5
        ax.plot(
            [i + offset, i + offset],
            [0 + offset, N + offset],
            color="black",
            linewidth=0.5,
        )
        ax.plot(
            [0 + offset, N + offset],
            [i + offset, i + offset],
            color="black",
            linewidth=0.5,
        )

    # Draw policy
    for i in range(N):
        for j in range(N):
            center_x = j + 0.5
            center_y = N - i - 0.5

            for action_idx, prob in enumerate(policy[N * i + j]):
                if prob > 0:
                    delta = deltas[action_idx]
                    if tuple(delta) == (0, 0):
                        radius = 0.2 * prob.item()
                        circle = plt.Circle(
                            (center_x, center_y),
                            radius,
                            color="red",
                            alpha=prob.item(),
                        )
                        ax.add_patch(circle)
                        continue

                    di, dj = delta * 0.4
                    dx, dy = dj, -di
                    ax.arrow(
                        center_x - dx / 2,
                        center_y - dy / 2,
                        dx,
                        dy,
                        head_width=0.2,
                        head_length=0.2,
                        fc="blue",
                        ec="blue",
                        alpha=prob.item(),
                    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0.5, N + 0.5))
    ax.set_yticks(np.arange(0.5, N + 0.5))
    ax.set_xticklabels(np.arange(N))
    ax.set_yticklabels(np.arange(N))


def imshow(values: torch.Tensor, ax: plt.Axes, grid_size: int):
    global_min = values.min().item()
    global_max = values.max().item()

    values = values.reshape((grid_size, grid_size))
    im = ax.imshow(
        values, cmap="hot", interpolation="nearest", vmin=global_min, vmax=global_max
    )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# whitelist
GridWorld.visualize_policy
GridWorld.visualize_values
GridWorld.create_exploration_policy
