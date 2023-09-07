import math
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


def visualize_values(
    grid_size: int, n_rounds: int, V: torch.Tensor, policy_idx: int = 0
):
    global_min = V[:, policy_idx].min().item()
    global_max = V[:, policy_idx].max().item()
    fig, axs = plt.subplots(n_rounds, 1, figsize=(5, 5 * n_rounds))

    for k in range(n_rounds):
        values = V[k, policy_idx, :-1].reshape((grid_size, grid_size))
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


def compute_policy_towards_goal(states, goals, grid_size):
    # Expand goals and states for broadcasting
    expanded_goals = goals[:, None, :]
    expanded_states = states[None, :, :]

    # Calculate the difference between each state and the goals
    diff = expanded_goals - expanded_states

    # Determine the action indices to move toward the goal for each state
    positive_x_i, positive_x_j = (diff[..., 0] > 0).nonzero(as_tuple=True)
    negative_x_i, negative_x_j = (diff[..., 0] < 0).nonzero(as_tuple=True)
    positive_y_i, positive_y_j = (diff[..., 1] > 0).nonzero(as_tuple=True)
    negative_y_i, negative_y_j = (diff[..., 1] < 0).nonzero(as_tuple=True)
    equal_i, equal_j = (diff == 0).all(-1).nonzero(as_tuple=True)

    # Initialize the actions tensor
    n_states = grid_size**2 + 1
    absorbing_state_idx = n_states - 1
    actions = torch.zeros(goals.size(0), n_states, 4)

    actions[positive_x_i, positive_x_j, 1] = 1  # Move down
    actions[negative_x_i, negative_x_j, 0] = 1  # Move up
    actions[positive_y_i, positive_y_j, 3] = 1  # Move right
    actions[negative_y_i, negative_y_j, 2] = 1  # Move left
    actions[equal_i, equal_j, :] = 1  # Random Movement
    actions[:, absorbing_state_idx, :] = 1  # Random Movement

    # Normalize using softmax along the action dimension
    return actions / actions.sum(-1, keepdim=True)


def value_iteration(grid_size: int, n_rounds: int, n_steps: int):
    deltas = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
    B = n_steps
    N = grid_size**2 + 1
    A = len(deltas)
    goals = torch.randint(0, grid_size, (n_steps, 2))
    states = torch.tensor([[i, j] for i in range(grid_size) for j in range(grid_size)])
    Pi = compute_policy_towards_goal(states, goals, grid_size)
    assert [*Pi.shape] == [B, N, A]

    # Compute next states for each action and state for each batch (goal)
    next_states = states[:, None] + deltas[None, :]
    next_states = torch.clamp(next_states, 0, grid_size - 1)
    S_ = next_states[..., 0] * grid_size + next_states[..., 1]  # Convert to indices

    # Determine if next_state is the goal for each batch (goal)
    is_goal = (goals[:, None] == states[None]).all(-1)

    # Modify transition to go to absorbing state if the next state is a goal
    absorbing_state_idx = N - 1
    S_ = S_[None].tile(B, 1, 1)
    S_[is_goal[..., None].expand_as(S_)] = absorbing_state_idx

    # Insert row for absorbing state
    padding = (0, 0, 0, 1)  # left 0, right 0, top 0, bottom 1
    S_ = F.pad(S_, padding, value=absorbing_state_idx)
    T = F.one_hot(S_, num_classes=N).float()
    R = is_goal.float()[..., None].tile(1, 1, A)
    R = F.pad(R, padding, value=0)  # Insert row for absorbing state

    # Compute the policy conditioned transition function
    Pi_ = Pi.view(B * N, 1, A)
    T_ = T.view(B * N, A, N)
    T_Pi = torch.bmm(Pi_, T_)
    T_Pi = T_Pi.view(B, N, N)

    gamma = 0.99  # Assuming a discount factor

    # Initialize V_0
    V = torch.zeros((n_rounds, n_steps, N), dtype=torch.float32)
    for k in tqdm(range(n_rounds - 1)):
        ER = (Pi * R).sum(-1)
        EV = (T_Pi * V[k, :, None]).sum(-1)
        V[k + 1] = ER + gamma * EV

    # visualize_values(grid_size, n_rounds, V, policy_idx=0)
    return goals, R, V


def round_to(tensor, decimals=2):
    return (tensor * 10**decimals).round() / (10**decimals)


def quantize_tensor(tensor, n_bins):
    # Flatten tensor
    flat_tensor = tensor.flatten()

    # Sort the flattened tensor
    sorted_tensor, _ = torch.sort(flat_tensor)

    # Determine the thresholds for each bin
    n_points_per_bin = int(math.ceil(len(sorted_tensor) / n_bins))
    thresholds = sorted_tensor[::n_points_per_bin].contiguous()

    # Assign each value in the flattened tensor to a bucket
    # The bucket number is the quantized value
    quantized_tensor = torch.bucketize(flat_tensor, thresholds)

    # Reshape the quantized tensor to the original tensor's shape
    quantized_tensor = quantized_tensor.view(tensor.shape)

    return quantized_tensor


class RLData(Dataset):
    def __init__(
        self,
        grid_size: int,
        min_order: int,
        max_order: int,
        n_bins: int,
        n_steps: int,
        seq_len: int,
    ):
        n_rounds = 2 * grid_size
        goals, R, V = value_iteration(
            grid_size=grid_size, n_steps=n_steps, n_rounds=n_rounds
        )

        def get_indices(states: torch.Tensor):
            # Step 1: Flatten states
            states_flattened = states.view(-1, 2)

            # Step 2: Convert states to indices
            indices = states_flattened[:, 0] * grid_size + states_flattened[:, 1]

            # Step 3: Reshape indices
            indices_reshaped = indices.view(n_steps, seq_len)

            # Step 4: Use advanced indexing
            return torch.arange(n_steps)[:, None], indices_reshaped

        states = torch.randint(0, grid_size, (n_steps, seq_len, 2))
        mapping = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
        actions = torch.randint(0, 4, (n_steps, seq_len))
        deltas = mapping[actions]
        next_states = torch.clamp(states + deltas, 0, grid_size - 1)
        idxs1, idxs2 = get_indices(states)
        rewards = R[idxs1, idxs2].gather(dim=2, index=actions[..., None])

        if max_order is None:
            max_order = len(V) - 2
        if min_order is None:
            min_order = 0
        order = torch.randint(min_order, max_order + 1, (n_steps, seq_len))
        idxs1, idxs2 = get_indices(next_states)

        V1 = V[order, idxs1, idxs2]
        V2 = V[order + 1, idxs1, idxs2]
        print("Computing unique values...", end="", flush=True)
        V1 = quantize_tensor(V1, n_bins)
        print("✓", end="", flush=True)
        V2 = quantize_tensor(V2, n_bins)
        print("✓")
        self.X = (
            torch.cat(
                [states, actions[..., None], rewards, order[..., None], V1[..., None]],
                -1,
            )
            .long()
            .cuda()
        )

        self.Z = V2.cuda()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx]
