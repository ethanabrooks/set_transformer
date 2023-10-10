import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from discretization import contiguous_integers, round_tensor
from metrics import LossType


class RLData(Dataset):
    def __init__(
        self,
        grid_size: int,
        loss_type: LossType,
        n_pi_bins: int,
        n_policies: int,
        n_v1_bins: int,
        order_delta: int,
    ):
        n_rounds = 2 * grid_size

        # 2D deltas for up, down, left, right
        deltas = torch.tensor([[0, 1], [0, -1], [-1, 0], [1, 0]])
        A = len(deltas)
        G = grid_size**2  # number of goals
        N = G + 1  # number of states - absorbing state
        P = n_policies

        all_states_1d = torch.arange(grid_size)
        all_states = torch.stack(
            torch.meshgrid(all_states_1d, all_states_1d, indexing="ij"), -1
        ).reshape(-1, 2)
        assert [*all_states.shape] == [G, 2]

        # get next states for product of states and actions
        next_states = all_states[:, None] + deltas[None]
        assert [*next_states.shape] == [G, A, 2]
        next_states = torch.clamp(next_states, 0, grid_size - 1)  # stay in bounds

        # add absorbing state for goals
        goal_idxs = torch.randint(0, G, (P,))
        next_state_idxs = next_states[..., 0] * grid_size + next_states[..., 1]
        # add absorbing state
        next_state_idxs = F.pad(next_state_idxs, (0, 0, 0, 1), value=G)
        is_goal = next_state_idxs[None] == goal_idxs[:, None, None]
        # transition to absorbing state instead of goal
        next_state_idxs = next_state_idxs[None].tile(P, 1, 1)
        next_state_idxs[is_goal] = G

        T: torch.Tensor = F.one_hot(next_state_idxs, num_classes=N)  # transition matrix
        R = is_goal.float()  # reward function

        alpha = torch.ones(A)
        Pi = torch.distributions.Dirichlet(alpha).sample((P, N))  # random policies
        assert [*Pi.shape] == [P, N, A]

        # Compute the policy conditioned transition function
        Pi = round_tensor(Pi, n_pi_bins) / n_pi_bins
        Pi = Pi.float()
        Pi = Pi / Pi.sum(-1, keepdim=True)
        Pi_ = Pi.view(P * N, 1, A)
        T_ = T.float().view(P * N, A, N)
        T_Pi = torch.bmm(Pi_, T_)
        T_Pi = T_Pi.view(P, N, N)

        gamma = 1  # discount factor

        # Initialize V_0
        V1 = torch.zeros((n_rounds - 1, n_policies, N), dtype=torch.float)
        V2 = torch.zeros((n_rounds, n_policies, N), dtype=torch.float)

        for k in tqdm(range(n_rounds - 1)):  # n_rounds of policy evaluation
            ER = (Pi * R).sum(-1)
            V1[k] = round_tensor(V2[k], n_v1_bins) / n_v1_bins
            EV = (T_Pi * V1[k, :, None]).sum(-1)
            Vk1 = ER + gamma * EV
            V2[k + 1] = Vk1

        states = torch.arange(N).repeat_interleave(A)
        states = states[None].tile(n_policies, 1)
        actions = torch.arange(A).repeat(N)
        actions = actions[None].tile(n_policies, 1)

        transition_probs = T[torch.arange(P)[:, None], states, actions]
        next_states = transition_probs.argmax(-1)

        idxs1 = torch.arange(n_policies)[:, None]
        idxs2 = states

        # Gather probabilities from Pi that correspond to states
        _action_probs = Pi[idxs1, idxs2]
        rewards = R[idxs1, idxs2].gather(dim=2, index=actions[..., None])

        # sample order -- number of steps of policy evaluation
        order = torch.randint(0, len(V1) - 1, (P, 1)).tile(1, A * N)

        _V1 = V1[order, idxs1, idxs2]
        order2 = torch.clamp(order + order_delta, 0, len(V2) - 1)
        _V2 = V2[order2, idxs1, idxs2]

        # discretize continuous values
        action_probs, self.decode_action_probs = contiguous_integers(_action_probs)
        assert torch.equal(_action_probs, self.decode_action_probs[action_probs])
        unique: torch.Tensor = _V2.unique()
        V1 = unique.numel() * _V1  # without this, sinusoidal encoding does not work.
        self.loss_type = loss_type
        if loss_type == LossType.MSE:
            V2 = _V2
        elif loss_type == LossType.CROSS_ENTROPY:
            V2, self.decode_V = contiguous_integers(_V2)
            assert torch.equal(_V2, self.decode_V[V2])
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        continuous = [
            action_probs,
            _V1[..., None],
        ]
        discrete = [
            states[..., None],
            actions[..., None],
            next_states[..., None],
            rewards,
        ]
        self.continuous = torch.cat(continuous, -1).long().cuda()
        self.discrete = torch.cat(discrete, -1).long().cuda()

        self.targets = V2.cuda()

    @property
    def max(self):
        _, _, d = self.continuous.shape
        return self.continuous.view(-1, d).max(0).values

    @property
    def min(self):
        _, _, d = self.continuous.shape
        return self.continuous.view(-1, d).min(0).values

    def __len__(self):
        return len(self.discrete)

    def __getitem__(self, idx):
        return self.continuous[idx], self.discrete[idx], self.targets[idx]

    @property
    def decode_outputs(self):
        if self.loss_type == LossType.MSE:
            return None
        else:
            return self.decode_V
