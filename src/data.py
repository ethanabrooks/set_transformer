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
        V = torch.zeros((n_rounds, n_policies, N), dtype=torch.float)
        self.V = V

        for k in tqdm(range(n_rounds - 1)):  # n_rounds of policy evaluation
            ER = (Pi * R).sum(-1)
            V[k] = V[k]
            EV = (T_Pi * V[k, :, None]).sum(-1)
            Vk1 = ER + gamma * EV
            V[k + 1] = Vk1

        states = torch.arange(N).repeat_interleave(A)
        states = states[None].tile(n_policies, 1)
        actions = torch.arange(A).repeat(N)
        actions = actions[None].tile(n_policies, 1)

        transition_probs = T[torch.arange(P)[:, None], states, actions]
        next_states = transition_probs.argmax(-1)

        idxs1 = torch.arange(n_policies)[:, None]
        idxs2 = states

        # Gather probabilities from Pi that correspond to states
        action_probs = Pi[idxs1, idxs2]
        rewards = R[idxs1, idxs2].gather(dim=2, index=actions[..., None])

        # sample order -- number of steps of policy evaluation
        input_order = torch.randint(0, self.max_order, (P, 1)).tile(1, A * N)

        order = [input_order + o for o in range(len(V))]
        order = [torch.clamp(o, 0, self.max_order) for o in order]
        V = [V[o, idxs1, idxs2] for o in order]

        self.loss_type = loss_type
        if loss_type == LossType.MSE:
            pass
        elif loss_type == LossType.CROSS_ENTROPY:
            _V, self.decode_V = contiguous_integers(V)
            assert torch.equal(V, self.decode_V[_V])
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        self.values = [torch.Tensor(v).cuda() for v in V]
        self.action_probs = torch.Tensor(action_probs).cuda()
        discrete = [
            states[..., None],
            actions[..., None],
            next_states[..., None],
            rewards,
        ]
        self.discrete = torch.cat(discrete, -1).long().cuda()

    def __len__(self):
        return len(self.discrete)

    def __getitem__(self, idx):
        return (
            self.action_probs[idx],
            self.discrete[idx],
            *[v[idx] for v in self.values],
        )

    @property
    def decode_outputs(self):
        if self.loss_type == LossType.MSE:
            return None
        else:
            return self.decode_V

    @property
    def max_order(self):
        return len(self.V) - 1
