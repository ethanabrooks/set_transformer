import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from discretization import round_tensor


class RLData(Dataset):
    def __init__(
        self,
        grid_size: int,
        include_v1: bool,
        min_order: int,
        max_order: int,
        n_input_bins: int,
        n_output_bins: int,
        n_policies: int,
        n_steps: int,
    ):
        n_rounds = 2 * grid_size

        deltas = torch.tensor([-1, 1])  # 1D deltas (left and right)
        B = n_steps
        N = grid_size + 1
        A = len(deltas)
        goals = torch.randint(0, grid_size, (n_steps,))
        all_states = torch.arange(grid_size + 1)  # +1 for absorbing state
        alpha = torch.ones(A)
        if n_policies is None:
            n_policies = B
        Pi = (
            torch.distributions.Dirichlet(alpha)  # random policies
            .sample((n_policies, N))
            .tile(math.ceil(B / n_policies), 1, 1)[:B]
        )
        assert [*Pi.shape] == [B, N, A]

        # get next states for product of states and actions
        next_states = all_states[..., None] + deltas[None]
        assert [*next_states.shape] == [N, A]
        next_states = torch.clamp(next_states, 0, grid_size - 1)  # stay in bounds
        next_states = next_states[None].tile(B, 1, 1)
        assert [*next_states.shape] == [B, N, A]
        is_goal = all_states == goals[:, None]
        # transition to absorbing state instead of goal
        next_states[is_goal] = grid_size
        # absorbing state transitions to itself
        next_states[:, grid_size] = grid_size

        T = F.one_hot(next_states, num_classes=N).float()  # transition matrix
        R = is_goal.float()[..., None].tile(1, 1, A)  # reward function

        # Compute the policy conditioned transition function
        Pi = round_tensor(Pi, n_input_bins).float()
        Pi_ = Pi.view(B * N, 1, A)
        T_ = T.view(B * N, A, N)
        T_Pi = torch.bmm(Pi_, T_)
        T_Pi = T_Pi.view(B, N, N)

        gamma = 1  # discount factor

        # Initialize V_0
        V = torch.zeros((n_rounds, n_steps, N), dtype=torch.float)

        for k in tqdm(range(n_rounds - 1)):  # n_rounds of policy evaluation
            ER = (Pi * R).sum(-1)
            EV = (T_Pi * V[k, :, None]).sum(-1)
            Vk1 = ER + gamma * EV
            V[k + 1] = Vk1

        # Gather probabilities from Pi that correspond to states
        states = all_states[..., None].tile(n_steps, 1, A).reshape(n_steps, -1)
        probabilities = Pi[torch.arange(n_steps)[:, None], states]

        # action indices corresponding to states, next_states
        actions = (
            torch.arange(A)[None]
            .expand(len(all_states), -1)
            .reshape(-1)
            .expand(n_steps, -1)
        )

        def get_indices(states: torch.Tensor):
            # In 1D, the state itself is the index
            return torch.arange(n_steps)[:, None], states

        # rewards corresponding to states (reward is not action dependent)
        idxs1, idxs2 = get_indices(states)
        rewards = R[idxs1, idxs2].gather(dim=2, index=actions[..., None])

        if max_order is None:
            max_order = len(V) - 2
        if min_order is None:
            min_order = 0

        # sample order -- number of steps of policy evaluation
        seq_len = A * len(all_states)
        order = torch.randint(min_order, max_order + 1, (n_steps, 1)).tile(1, seq_len)

        idxs1, idxs2 = get_indices(states)
        V1 = V[order, idxs1, idxs2]
        V2 = V[order + 1, idxs1, idxs2]

        # discretize continuous values
        V1 = round_tensor(V1, n_input_bins, contiguous=True)
        V2 = round_tensor(V2, n_output_bins, contiguous=True)
        probabilities = round_tensor(probabilities, n_input_bins, contiguous=True)

        self.X = (
            torch.cat(
                [
                    states[..., None],
                    probabilities,
                    actions[..., None],
                    next_states.view(B, N * A, 1),
                    rewards,
                    *([V1[..., None]] if include_v1 else []),
                ],
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
