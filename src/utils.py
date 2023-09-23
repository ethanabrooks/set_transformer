import math

import torch
import torch.nn.functional as F
from tqdm import tqdm


def policy_evaluation(grid_size: int, n_policies: int, n_rounds: int, n_steps: int):
    deltas = torch.tensor([-1, 1])  # 1D deltas (left and right)
    B = n_steps
    N = grid_size + 1
    A = len(deltas)
    goals = torch.randint(0, grid_size, (n_steps,))
    states = torch.arange(grid_size)
    alpha = torch.ones(A)
    if n_policies is None:
        n_policies = B
    Pi = (
        torch.distributions.Dirichlet(alpha)
        .sample((n_policies, N))
        .tile(math.ceil(B / n_policies), 1, 1)[:B]
    )
    assert [*Pi.shape] == [B, N, A]

    # Compute next states for each action for each batch (goal)
    next_states = states[:, None] + deltas[None, :]
    next_states = torch.clamp(next_states, 0, grid_size - 1)
    S_ = next_states

    # Determine if next_state is the goal for each batch (goal)
    is_goal = goals[:, None] == states[None]

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
    return Pi, R, V, goals


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

    # Make the quantized values contiguous
    unique_bins = torch.unique(quantized_tensor)
    for i, bin in enumerate(unique_bins):
        quantized_tensor[quantized_tensor == bin] = i

    # Reshape the quantized tensor to the original tensor's shape
    quantized_tensor = quantized_tensor.view(tensor.shape)

    return quantized_tensor


def round_tensor(tensor, round_to):
    discretized = (tensor * round_to).round().long()

    # Make the quantized values contiguous
    unique_bins = torch.unique(discretized)
    for i, bin in enumerate(unique_bins):
        discretized[discretized == bin] = i

    # Reshape the quantized tensor to the original tensor's shape
    discretized = discretized.view(tensor.shape)

    return discretized


if __name__ == "__main__":
    quantize_tensor  # whitelist
