from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.base import Step


class GridWorld:
    def __init__(
        self,
        absorbing_state: bool,
        dense_reward: bool,
        gamma: float,
        grid_size: int,
        heldout_goals: list[tuple[int, int]],
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
        T = n_tasks

        # add absorbing state for goals
        self.absorbing_state = absorbing_state
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

        is_wall = torch.rand(T, G, A, 1) < p_wall
        goal_idxs = torch.randint(0, G, (T,))
        self.goals = torch.stack([goal_idxs // grid_size, goal_idxs % grid_size], dim=1)

        # Compute next states for each action and state for each batch (goal)
        next_states = self.states[:, None] + self.deltas[None, :]
        assert [*next_states.shape] == [G, A, 2]
        states = self.states[None, :, None].tile(T, 1, A, 1)
        next_states = next_states[None].tile(T, 1, 1, 1)
        next_states = states * is_wall + next_states * (~is_wall)
        next_states = torch.clamp(next_states, 0, grid_size - 1)  # stay in bounds
        next_state_indices = (
            next_states[..., 0] * self.grid_size + next_states[..., 1]
        )  # Convert to indices

        # Determine if next_state is the goal for each batch (goal)
        # is_goal = (self.goals[:, None] == self.states[None]).all(-1)
        is_goal = goal_idxs[:, None, None] == next_state_indices

        # Modify transition to go to absorbing state if the next state is a goal
        absorbing_state_idx = self.n_states - 1
        if self.absorbing_state:
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
        if self.absorbing_state:
            self.R = F.pad(R, padding, value=0)  # Insert row for absorbing state

    @property
    def n_states(self):
        n_states = self.grid_size**2
        if self.absorbing_state:
            n_states += 1
        return n_states

    # def check_actions(self, actions: torch.Tensor):
    #     B = self.n_tasks
    #     A = len(self.deltas)
    #     assert [*actions.shape] == [B]
    #     assert actions.max() < A
    #     assert 0 <= actions.min()

    def check_pi(self, Pi: torch.Tensor):
        B = self.n_tasks
        N = self.n_states
        A = len(self.deltas)
        assert [*Pi.shape] == [B, N, A]

    # def check_states(self, states: torch.Tensor):
    #     B = self.n_tasks
    #     assert [*states.shape] == [B]
    #     assert states.max() < self.n_states
    #     assert 0 <= states.min()

    # def check_time_step(self, time_step: torch.Tensor):
    #     B = self.n_tasks
    #     assert [*time_step.shape] == [B]

    def check_V(self, V: torch.Tensor):
        B = self.n_tasks
        N = self.n_states
        assert [*V.shape] == [B, N]

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
        if self.absorbing_state:
            # Insert row for absorbing state
            policy = F.pad(policy, (0, 0, 0, 1), value=0)
            policy[-1, 0] = 1  # last state is terminal
        # self.visualize_policy(policy[None].tile(self.n_tasks, 1, 1))
        return policy

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
        rewards = torch.zeros((B, trajectory_length))
        done = torch.zeros((B, trajectory_length), dtype=torch.bool)
        current_states = self.reset_fn()
        time_step = torch.zeros((B))

        for t in tqdm(range(trajectory_length), desc="Sampling trajectories"):
            # Convert current current_states to indices
            current_state_indices = (
                current_states[:, 0] * self.grid_size + current_states[:, 1]
            )

            # Sample actions from the policy
            A = (
                torch.multinomial(Pi[torch.arange(B), current_state_indices], 1)
                .squeeze(1)
                .long()
            )

            # Convert current current_states to indices
            current_state_indices = states[:, 0] * self.grid_size + states[:, 1]
            next_state_indices, R, D, _ = self.step_fn(
                current_state_indices, A, time_step
            )

            # Convert next state indices to coordinates
            next_states = torch.stack(
                (
                    next_state_indices // self.grid_size,
                    next_state_indices % self.grid_size,
                ),
                dim=1,
            )
            next_states_on_reset = self.reset_fn()
            next_states[D] = next_states_on_reset[D]

            # Store the current current_states and rewards
            states[:, t] = current_states
            actions[:, t] = A
            rewards[:, t] = R
            done[:, t] = D
            time_step += 1
            time_step[D] = 0

            # Update current current_states
            current_states = next_states

        return (
            Step(
                tasks=self.goals[:, None].expand_as(states),
                observations=states,
                actions=actions[..., None],
                rewards=rewards,
            ),
            done,
        )

    def policy_evaluation(self, Pi: torch.Tensor, V: torch.Tensor = None):
        self.check_pi(Pi)

        B = self.n_tasks
        N = self.n_states
        A = len(self.deltas)
        assert [*Pi.shape] == [B, N, A]

        # Compute the policy conditioned transition function
        Pi_ = Pi.view(B * N, 1, A)
        T_ = self.T.view(B * N, A, N)
        T_Pi = torch.bmm(Pi_, T_)
        T_Pi = T_Pi.view(B, N, N)

        # Initialize V_0
        if V is None:
            V = torch.zeros((B, N), dtype=torch.float32)
        self.check_V(V)
        ER = (Pi * self.R).sum(-1)
        EV = (T_Pi * V[:, None]).sum(-1)
        V = ER + self.gamma * EV
        return V

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

        arange = torch.arange(self.n_tasks)[:, None].expand_as(states).flatten()
        states = states.flatten()
        actions = actions.flatten()

        rewards = self.R[arange, states, actions]
        rewards = rewards.reshape(shape)

        # Compute next state indices
        next_states = torch.argmax(self.T[arange, states, actions], dim=1)
        next_states = next_states.reshape(shape)

        if time_step is None:
            done = torch.zeros_like(states, dtype=torch.bool)
        elif episode_length is not None:
            done = time_step + 1 == episode_length
        else:
            raise ValueError("Either episode_length or time_step must be provided")
        if self.terminate_on_goal:
            done = done | (states == self.goals).all(-1)
        return next_states, rewards, done, {}

    def visualize_policy(self, Pi, task_idx: int = 0):
        N = self.grid_size
        policy = Pi[task_idx]
        _, ax = plt.subplots(figsize=(6, 6))
        plt.xlim(0, N)  # Adjusted the starting limit
        plt.ylim(0, N)

        # Draw grid
        for i in range(N + 1):
            offset = 0.5
            plt.plot(
                [i + offset, i + offset],
                [0 + offset, N + offset],
                color="black",
                linewidth=0.5,
            )
            plt.plot(
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
                    if (
                        prob > 0
                    ):  # Only draw if there's a non-zero chance of taking the action
                        delta = self.deltas[action_idx]
                        if tuple(delta) == (0, 0):
                            radius = (
                                0.2 * prob.item()
                            )  # The circle's radius could be proportional to the action probability
                            circle = plt.Circle(
                                (center_x, center_y),
                                radius,
                                color="red",
                                alpha=prob.item(),
                            )
                            ax.add_patch(circle)
                            continue  # Skip the arrow drawing part for no-op action

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
                            alpha=prob.item(),  # Set the opacity according to the probability
                        )

        plt.gca().set_aspect("equal", adjustable="box")
        plt.xticks(np.arange(0.5, N + 0.5), np.arange(N))
        plt.yticks(np.arange(0.5, N + 0.5), np.arange(N))
        plt.savefig("policy.png")


# whitelist
GridWorld.visualize_policy
GridWorld.create_exploration_policy
GridWorld.get_trajectories
