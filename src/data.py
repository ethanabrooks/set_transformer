import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from discretization import contiguous_integers, round_tensor


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
        Pi = round_tensor(Pi, n_pi_bins)
        Pi = Pi.float()
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
        _action_probs = Pi[idxs1, idxs2]
        rewards = R[idxs1, idxs2].gather(dim=2, index=actions[..., None])

        # sample order -- number of steps of policy evaluation
        order = torch.randint(0, len(V) - 1, (P, 1)).tile(1, A * N)

        _V1 = round_tensor(V[order, idxs1, idxs2], n_v1_bins)
        _V2 = round_tensor(V[order + 1, idxs1, idxs2], n_v2_bins)

        # discretize continuous values
        action_probs, self.decode_action_probs = contiguous_integers(_action_probs)
        assert torch.equal(_action_probs, self.decode_action_probs[action_probs])
        V1, self.decode_V1 = contiguous_integers(_V1)
        assert torch.equal(_V1, self.decode_V1[V1])
        V2, self.decode_V2 = contiguous_integers(_V2)
        assert torch.equal(_V2, self.decode_V2[V2])

        X = [
            states[..., None],
            action_probs,
            actions[..., None],
            next_states[..., None],
            rewards,
            V1[..., None],
        ]
        self.shapes = [x.shape for x in X]
        self.X = torch.cat(X, -1).long().cuda()

        self.Z = V2.cuda()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx]

    def decode_inputs(self, X: torch.Tensor):
        dims = [shape[-1] for shape in self.shapes]
        states, action_probs, actions, next_states, rewards, V1 = torch.split(
            X, dims, dim=-1
        )
        action_probs = self.decode_action_probs[action_probs]
        V1 = self.decode_V1[V1]
        return states, action_probs, actions, next_states, rewards, V1

    def decode_outputs(self, V2: torch.Tensor):
        return self.decode_V2[V2]


# whitelist
RLData.decode_inputs
RLData.decode_outputs
