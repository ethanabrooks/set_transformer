import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from discretization import round_tensor


class RLData(Dataset):
    def __init__(
        self,
        grid_size: int,
        n_pi_bins: int,
        n_v1_bins: int,
        n_v2_bins: int,
        n_policies: int,
    ):
        n_rounds = 2 * grid_size

        deltas = torch.tensor([-1, 1])  # 1D deltas (left and right)
        A = len(deltas)
        P = n_policies
        N = grid_size + 1
        all_states = torch.arange(grid_size + 1)  # +1 for absorbing state

        # get next states for product of states and actions
        next_states = all_states[..., None] + deltas[None]
        assert [*next_states.shape] == [N, A]
        next_states = torch.clamp(next_states, 0, grid_size - 1)  # stay in bounds

        # send to absorbing state if goal is reached
        goals = torch.randint(0, grid_size, (n_policies,))  # random goal per policy
        next_states = next_states[None].tile(P, 1, 1)
        assert [*next_states.shape] == [P, N, A]
        is_goal = all_states == goals[:, None]
        # transition to absorbing state instead of goal
        next_states[is_goal] = grid_size
        # absorbing state transitions to itself
        next_states[:, grid_size] = grid_size

        T: torch.Tensor = F.one_hot(next_states, num_classes=N)  # transition matrix
        R = is_goal.float()[..., None].tile(1, 1, A)  # reward function

        alpha = torch.ones(A)
        Pi = torch.distributions.Dirichlet(alpha).sample((P, N))  # random policies
        assert [*Pi.shape] == [P, N, A]

        # Compute the policy conditioned transition function
        Pi = round_tensor(Pi, n_pi_bins).float() / n_pi_bins
        Pi_ = Pi.view(P * N, 1, A)
        T_ = T.float().view(P * N, A, N)
        T_Pi = torch.bmm(Pi_, T_)
        T_Pi = T_Pi.view(P, N, N)

        gamma = 1  # discount factor

        # Initialize V_0
        V = torch.zeros((n_rounds, n_policies, N), dtype=torch.float)

        for k in tqdm(range(n_rounds - 1)):  # n_rounds of policy evaluation
            ER = (Pi * R).sum(-1)
            EV = (T_Pi * V[k, :, None]).sum(-1)
            Vk1 = ER + gamma * EV
            V[k + 1] = round_tensor(Vk1, n_v1_bins).float() / n_v1_bins

        states = all_states.repeat_interleave(A)
        states = states[None].tile(n_policies, 1)
        actions = torch.arange(A).repeat(N)
        actions = actions[None].tile(n_policies, 1)

        idxs1 = torch.arange(n_policies)[:, None]
        idxs2 = states

        # Gather probabilities from Pi that correspond to states
        action_probs = Pi[idxs1, idxs2]
        rewards = R[idxs1, idxs2].gather(dim=2, index=actions[..., None])

        # sample order -- number of steps of policy evaluation
        order = torch.randint(0, len(V) - 1, (n_policies, 1)).tile(1, A * N)

        V1 = V[order, idxs1, idxs2]
        V2 = V[order + 1, idxs1, idxs2]

        # discretize continuous values
        action_probs = round_tensor(action_probs, n_pi_bins, contiguous=True)
        V1 = round_tensor(V1, n_v1_bins, contiguous=True)
        V2 = round_tensor(V2, n_v2_bins, contiguous=True)

        self.X = (
            torch.cat(
                [
                    states[..., None],
                    action_probs,
                    actions[..., None],
                    next_states.view(P, N * A, 1),
                    rewards,
                    V1[..., None],
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
