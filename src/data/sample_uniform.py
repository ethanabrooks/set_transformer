import itertools

import torch
import torch.nn.functional as F

import data.base
from metrics import compute_rmse


def round_tensor(tensor: torch.Tensor, round_to: int):
    return (tensor * round_to).round().long()


class RLData(data.base.RLData):
    def __init__(
        self,
        grid_size: int,
        n_pi_bins: int,
        n_policies: int,
        omit_states_actions: int,
        p_wall: float,
        stop_at_rmse: float,
    ):
        # 2D deltas for up, down, left, right
        deltas = torch.tensor([[0, 1], [0, -1], [-1, 0], [1, 0]])
        A = len(deltas)
        G = grid_size**2  # number of goals
        N = G + 1  # number of goal states + absorbing state
        P = n_policies

        all_states_1d = torch.arange(grid_size)
        all_states = torch.stack(
            torch.meshgrid(all_states_1d, all_states_1d, indexing="ij"), -1
        ).reshape(-1, 2)
        assert [*all_states.shape] == [G, 2]

        # get next states for product of states and actions
        is_wall = torch.rand(P, G, A, 1) < p_wall
        next_states = all_states[:, None] + deltas[None]
        assert [*next_states.shape] == [G, A, 2]
        states = all_states[None, :, None].tile(P, 1, A, 1)
        next_states = next_states[None].tile(P, 1, 1, 1)
        next_states = states * is_wall + next_states * (~is_wall)
        next_states = torch.clamp(next_states, 0, grid_size - 1)  # stay in bounds

        # add absorbing state for goals
        goal_idxs = torch.randint(0, G, (P,))
        next_state_idxs = next_states[..., 0] * grid_size + next_states[..., 1]
        # add absorbing state
        next_state_idxs = F.pad(next_state_idxs, (0, 0, 0, 1), value=G)
        is_goal = next_state_idxs == goal_idxs[:, None, None]
        # transition to absorbing state instead of goal
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
        V = [torch.zeros((n_policies, N), dtype=torch.float)]

        print("Policy evaluation...")
        for k in itertools.count(1):  # n_rounds of policy evaluation
            ER = (Pi * R).sum(-1)
            Vk = V[-1]
            EV = (T_Pi * Vk[:, None]).sum(-1)
            Vk1 = ER + gamma * EV
            V.append(Vk1)
            rmse = compute_rmse(Vk1, Vk)
            print("Iteration:", k, "RMSE:", rmse)
            if rmse < stop_at_rmse:
                break
        V = torch.stack(V)
        self.V = V

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

        # sample n_bellman -- number of steps of policy evaluation
        input_n_bellman = torch.randint(0, self.max_n_bellman, (P, 1)).tile(1, A * N)

        n_bellman = [input_n_bellman + o for o in range(len(V))]
        n_bellman = [torch.clamp(o, 0, self.max_n_bellman) for o in n_bellman]
        V = [V[o, idxs1, idxs2] for o in n_bellman]

        self.values = [torch.Tensor(v) for v in V]
        self.action_probs = torch.Tensor(action_probs)
        discrete = [
            states[..., None],
            actions[..., None],
            next_states[..., None],
            rewards,
        ]
        self.discrete = torch.cat(discrete, -1).long()

        perm = torch.rand(P, N * A).argsort(dim=1)

        def shuffle(x: torch.Tensor):
            p = perm
            while p.dim() < x.dim():
                p = p[..., None]

            return torch.gather(x, 1, p.expand_as(x))

        *self.values, self.action_probs, self.discrete = [
            shuffle(x)[:, omit_states_actions:].cuda()
            for x in [*self.values, self.action_probs, self.discrete]
        ]
        self.input_n_bellman = input_n_bellman[:, omit_states_actions:].cuda()
