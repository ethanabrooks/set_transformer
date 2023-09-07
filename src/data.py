import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def visualize_values(
    grid_size: int, n_rounds: int, V: torch.Tensor, policy_idx: int = 0
):
    global_min = V[:, policy_idx].min().item()
    global_max = V[:, policy_idx].max().item()
    fig, axs = plt.subplots(n_rounds, 1, figsize=(5, 5 * n_rounds))

    for k in range(n_rounds):
        values = V[k, policy_idx].reshape((grid_size, grid_size))
        im = axs[k].imshow(
            values,
            cmap="hot",
            interpolation="nearest",
            vmin=global_min,
            vmax=global_max,
        )
        axs[k].set_title(f"Value function at iteration {k}")

        # Add colorbar to each subplot
        fig.colorbar(im, ax=axs[k], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(f"value_iteration{policy_idx}.png")


def value_iteration(grid_size: int, n_rounds: int, n_steps: int):
    n_states = grid_size**2

    goals = torch.randint(0, grid_size, (n_steps, 2))

    alpha = torch.ones(4)
    Pi = torch.distributions.Dirichlet(alpha).sample((n_steps, n_states))
    states = torch.tensor([[i, j] for i in range(grid_size) for j in range(grid_size)])
    deltas = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])

    # Compute next states for each action and state
    next_states = states[:, None] + deltas[None]

    # Ensure next states are within the grid boundaries
    next_states = torch.clamp(next_states, 0, grid_size - 1)
    S_ = next_states[:, :, 0] * grid_size + next_states[:, :, 1]
    R = (goals[:, None] == states[None]).all(-1)

    # Extend dimensions for broadcasting
    T = torch.zeros(n_steps, n_states, n_states)
    idx = torch.arange(n_states)[:, None]
    T[:, idx, S_] = Pi
    T_ = T.swapaxes(1, 2)

    gamma = 0.99  # Assuming a discount factor

    # Initialize V_0
    V = torch.zeros((n_rounds, n_steps, n_states), dtype=torch.float32)
    for k in tqdm(range(n_rounds - 1)):
        EV = torch.bmm(T_, V[k][..., None]).squeeze(-1)
        V[k + 1] = R + gamma * EV

    return goals, V


class RLData(Dataset):
    def __init__(self, grid_size, n_steps, seq_len):
        states = torch.randint(0, grid_size, (n_steps, seq_len, 2))
        goals = torch.randint(0, grid_size, (n_steps, 1, 2)).expand_as(states)
        actions = torch.randint(0, 4, (n_steps, seq_len))
        mapping = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
        deltas = mapping[actions]
        order = torch.randint(0, 2 * grid_size, (n_steps, seq_len))
        rewards = (states == goals).all(-1).long()
        Q = (states + deltas - goals).sum(-1).abs()
        Q_ = torch.min(Q, order)
        self.X = (
            torch.cat(
                [states, actions[..., None], rewards[..., None], Q_[..., None]], -1
            )
            .long()
            .cuda()
        )

        self.Z = torch.min(Q, order + 1).cuda()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx]
