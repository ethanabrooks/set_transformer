import itertools
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from dataset.utils import Transition
from metrics import compute_rmse
from tabular.maze import generate_maze, maze_to_state_action


class GridWorld:
    def __init__(
        self,
        absorbing_state: bool,
        dense_reward: bool,
        gamma: float,
        grid_size: int,
        heldout_goals: list[tuple[int, int]],
        n_maze: int,
        n_tasks: int,
        p_wall: float,
        seed: int,
        terminate_on_goal: bool,
        use_heldout_goals: bool,
    ):
        # transition to absorbing state instead of goal
        self.deltas = torch.tensor([[0, 1], [0, -1], [-1, 0], [1, 0]])
        A = len(self.deltas)
        G = grid_size**2  # number of goals
        M = n_maze
        T = n_tasks

        # add absorbing state for goals
        self.use_absorbing_state = absorbing_state
        self.dense_reward = dense_reward
        self.gamma = gamma
        self.grid_size = grid_size
        self.heldout_goals = heldout_goals
        self.random = np.random.default_rng(seed)
        self.terminate_on_goal = terminate_on_goal
        self.use_heldout_goals = use_heldout_goals

        self.states = torch.tensor(
            [[i, j] for i in range(grid_size) for j in range(grid_size)]
        )
        self.n_tasks = n_tasks

        # generate walls
        is_wall = torch.rand(T, G, A) < p_wall
        if n_maze:
            mazes = [
                maze_to_state_action(generate_maze(grid_size)).view(G, A)
                for _ in tqdm(range(M), desc="Generating mazes")
            ]
            mazes = torch.stack(mazes)
            assert [*mazes.shape] == [M, G, A]
            maze_idx = torch.randint(0, M, (T,))
            is_wall = mazes[maze_idx] & is_wall
            assert [*is_wall.shape] == [T, G, A]
        is_wall = is_wall[..., None]

        # Compute next states for each action and state for each batch (goal)
        next_states = self.states[:, None] + self.deltas[None, :]
        assert [*next_states.shape] == [G, A, 2]
        states = self.states[None, :, None].tile(T, 1, A, 1)
        next_states = next_states[None].tile(T, 1, 1, 1)
        next_states = states * is_wall + next_states * (~is_wall)
        next_states = torch.clamp(next_states, 0, grid_size - 1)  # stay in bounds
        next_state_indices = self.convert_2d_to_1d(next_states)  # Convert to indices

        # Determine if next_state is the goal for each batch (goal)
        # is_goal = (self.goals[:, None] == self.states[None]).all(-1)
        # create walls and goals
        self.goal_idxs = torch.randint(0, G, (T,))
        is_goal = self.goal_idxs[:, None, None] == next_state_indices

        # Modify transition to go to absorbing state if the next state is a goal
        absorbing_state_idx = self.n_states - 1
        if self.use_absorbing_state:
            # S_[is_goal[..., None].expand_as(S_)] = absorbing_state_idx
            next_state_indices[is_goal[:, :G]] = absorbing_state_idx

            # Insert row for absorbing state
            padding = (0, 0, 0, 1)  # left 0, right 0, top 0, bottom 1
            next_state_indices = F.pad(
                next_state_indices, padding, value=absorbing_state_idx
            )
        self.T: torch.Tensor = F.one_hot(next_state_indices, num_classes=self.n_states)
        self.T = self.T.float()

        if dense_reward:
            distance = (self.goals[:, None] - self.states[None]).abs().sum(-1)
            R = -distance.float()[..., None].tile(1, 1, len(self.deltas))
        else:
            R = is_goal.float()
        self.R = R
        if self.use_absorbing_state:
            self.R = F.pad(R, padding, value=0)  # Insert row for absorbing state

    @property
    def absorbing_state(self):
        return self.n_states - 1

    @property
    def goals(self):
        return self.convert_1d_to_2d(self.goal_idxs)

    @property
    def n_states(self):
        n_states = self.grid_size**2
        if self.use_absorbing_state:
            n_states += 1
        return n_states

    def convert_1d_to_2d(self, x: torch.Tensor):
        return torch.stack([x // self.grid_size, x % self.grid_size], dim=1)

    def convert_2d_to_1d(self, x: torch.Tensor):
        return self.grid_size * x[..., 0] + x[..., 1]

    # def check_actions(self, actions: torch.Tensor):
    #     B = self.n_tasks
    #     A = len(self.deltas)
    #     assert [*actions.shape] == [B]
    #     assert actions.max() < A
    #     assert 0 <= actions.min()

    # def check_pi(self, Pi: torch.Tensor):
    #     B = self.n_tasks
    #     N = self.n_states
    #     A = len(self.deltas)
    #     assert [*Pi.shape] == [B, N, A]

    # def check_states(self, states: torch.Tensor):
    #     B = self.n_tasks
    #     assert [*states.shape] == [B]
    #     assert states.max() < self.n_states
    #     assert 0 <= states.min()

    # def check_time_step(self, time_step: torch.Tensor):
    #     B = self.n_tasks
    #     assert [*time_step.shape] == [B]

    # def check_V(self, V: torch.Tensor):
    #     B = self.n_tasks
    #     N = self.n_states
    #     assert [*V.shape] == [B, N]

    def create_exploration_policy(self):
        N = self.grid_size
        A = len(self.deltas)

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
        idxs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Pi = torch.zeros_like(Q)
        Pi.scatter_(-1, Q.argmax(dim=-1, keepdim=True), 1.0)
        Q: torch.Tensor
        *_, Q = self.evaluate_policy_iteratively(
            Pi=Pi, stop_at_rmse=stop_at_rmse, idxs=idxs
        )
        return Q

    def evaluate_policy(
        self,
        Pi: torch.Tensor,
        Q: torch.Tensor = None,
        idxs: Optional[torch.Tensor] = None,
    ):
        # self.check_pi(Pi)

        B = self.n_tasks if idxs is None else len(idxs)
        N = self.n_states
        A = len(self.deltas)

        # Compute the policy conditioned transition function
        T = self.get_transitions(idxs).to(Pi.device)

        # Initialize Q_0
        if Q is None:
            Q = torch.zeros((B, N, A), dtype=torch.float32, device=Pi.device)
        R = self.get_rewards(idxs).to(Pi.device)
        assert [*R.shape] == [B, N, A]
        EQ = (Pi * Q).sum(-1)
        assert [*EQ.shape] == [B, N]
        EQ = (T * EQ[:, None, None]).sum(-1)
        assert [*EQ.shape] == [B, N, A]
        Q = R + self.gamma * EQ
        return Q

    def evaluate_policy_iteratively(
        self,
        Pi: torch.Tensor,
        stop_at_rmse: float,
        idxs: Optional[torch.Tensor] = None,
    ):
        B = self.n_tasks if idxs is None else len(idxs)
        S = self.n_states
        A = len(self.deltas)

        Q = [torch.zeros((B, S, A), device=Pi.device, dtype=torch.float)]
        for _ in itertools.count(1):  # n_rounds of policy evaluation
            Vk = Q[-1]
            Vk1 = self.evaluate_policy(Pi, Vk, idxs=idxs)
            Q.append(Vk1)
            rmse = compute_rmse(Vk1, Vk)
            # print("Iteration:", k, "RMSE:", rmse)
            if rmse < stop_at_rmse:
                break
        return Q

    def get_trajectories(
        self,
        episode_length: int,
        Pi: torch.Tensor,
        n_episodes: int = 1,
    ):
        B = self.n_tasks
        N = self.n_states
        A = len(self.deltas)
        assert [*Pi.shape] == [B, N, A]

        trajectory_length = episode_length * n_episodes
        states = torch.zeros((B, trajectory_length, 2), dtype=torch.int)
        actions = torch.zeros((B, trajectory_length), dtype=torch.int)
        action_probs = torch.zeros((B, trajectory_length, A), dtype=torch.float)
        next_states = torch.zeros((B, trajectory_length, 2), dtype=torch.int)
        rewards = torch.zeros((B, trajectory_length))
        done = torch.zeros((B, trajectory_length), dtype=torch.bool)
        S1 = self.reset_fn()
        time_step = torch.zeros((B))
        arange = torch.arange(B)

        for t in tqdm(range(trajectory_length), desc="Sampling trajectories"):
            # Convert current current_states to indices
            current_state_indices = self.convert_2d_to_1d(S1)

            # Sample actions from the policy
            A = (
                torch.multinomial(Pi[arange, current_state_indices], 1)
                .squeeze(1)
                .long()
            )

            # Convert current current_states to indices
            next_state_indices, R, D, _ = self.step_fn(
                states=current_state_indices,
                actions=A,
                episode_length=episode_length,
                time_step=time_step,
            )

            # Convert next state indices to coordinates
            S2 = self.convert_1d_to_2d(next_state_indices)

            # Store the current current_states and rewards
            states[:, t] = S1
            actions[:, t] = A
            action_probs[:, t] = Pi[arange, current_state_indices]
            next_states[:, t] = S2
            rewards[:, t] = R
            done[:, t] = D
            time_step += 1
            time_step[D] = 0

            # Update current current_states
            S1 = S2
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

    def get_rewards(self, idxs: Optional[torch.Tensor] = None):
        if idxs is None:
            return self.R
        return self.R.to(idxs.device)[idxs]

    def get_transitions(self, idxs: Optional[torch.Tensor] = None):
        if idxs is None:
            return self.T
        return self.T.to(idxs.device)[idxs]

    def reset_fn(self):
        array = self.random.choice(self.grid_size, size=(self.n_tasks, 2))
        return torch.tensor(array)

    def step_fn(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        episode_length: Optional[int] = None,
        time_step: Optional[torch.Tensor] = None,
    ):
        # self.check_states(states)
        # self.check_actions(actions)
        # self.check_time_step(time_step)
        assert len(states) == self.n_tasks
        assert states.shape == actions.shape
        shape = states.shape

        arange = torch.arange(self.n_tasks)
        while arange.dim() < states.dim():
            arange = arange[..., None]
        arange = arange.expand_as(states).flatten()
        states = states.flatten()
        actions = actions.flatten()

        rewards = self.R[arange, states, actions]

        # Compute next state indices
        next_states = torch.argmax(self.T[arange, states, actions], dim=1)

        if time_step is None:
            done = torch.zeros_like(states, dtype=torch.bool)
        elif episode_length is not None:
            done = time_step + 1 == episode_length
        else:
            raise ValueError("Either episode_length or time_step must be provided")
        if self.terminate_on_goal:
            done = done | (states == self.absorbing_state)
        next_states = next_states.reshape(shape)
        rewards = rewards.reshape(shape)
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
            values = values[..., :-1]
            im = ax.imshow(
                values.reshape((2 * self.grid_size, self.grid_size)),
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
GridWorld.get_trajectories
