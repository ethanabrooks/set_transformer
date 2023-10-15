import itertools

import torch

import data.base
from metrics import compute_rmse
from tabular.grid_world import GridWorld


def round_tensor(tensor: torch.Tensor, round_to: int):
    return (tensor * round_to).round().long()


class RLData(data.base.RLData):
    def __init__(
        self,
        grid_world_args: dict,
        n_pi_bins: int,
        n_data: int,
        omit_states_actions: int,
        seed: int,
        stop_at_rmse: float,
    ):
        # 2D deltas for up, down, left, right
        grid_world = GridWorld(**grid_world_args, n_tasks=n_data, seed=seed)
        A = len(grid_world.deltas)
        G = grid_world.grid_size**2  # number of goals
        S = G + 1  # number of goal states + absorbing state
        B = n_data

        alpha = torch.ones(A)
        Pi = torch.distributions.Dirichlet(alpha).sample((B, S))  # random policies
        assert [*Pi.shape] == [B, S, A]

        # Compute the policy conditioned transition function
        Pi = round_tensor(Pi, n_pi_bins) / n_pi_bins
        Pi = Pi.float()
        Pi = Pi / Pi.sum(-1, keepdim=True)

        # Initialize V_0
        V = [torch.zeros((B, S), dtype=torch.float)]

        print("Policy evaluation...")
        for k in itertools.count(1):  # n_rounds of policy evaluation
            Vk = V[-1]
            Vk1 = grid_world.policy_evaluation(Pi, Vk)
            V.append(Vk1)
            rmse = compute_rmse(Vk1, Vk)
            print("Iteration:", k, "RMSE:", rmse)
            if rmse < stop_at_rmse:
                break
        V = torch.stack(V)
        self.V = V

        states = torch.arange(S).repeat_interleave(A)
        states = states[None].tile(B, 1)
        actions = torch.arange(A).repeat(S)
        actions = actions[None].tile(B, 1)
        next_states, rewards, _, _ = grid_world.step_fn(states, actions)

        # sample n_bellman -- number of steps of policy evaluation
        input_n_bellman = torch.randint(0, self.max_n_bellman, (B, 1)).tile(1, A * S)

        n_bellman = [input_n_bellman + o for o in range(len(V))]
        n_bellman = [torch.clamp(o, 0, self.max_n_bellman) for o in n_bellman]
        arange = torch.arange(B)[:, None]
        action_probs = Pi[arange, states]
        V = [V[o, arange, states] for o in n_bellman]

        self.values = [torch.Tensor(v) for v in V]
        self.action_probs = torch.Tensor(action_probs)
        discrete = [
            states[..., None],
            actions[..., None],
            next_states[..., None],
            rewards[..., None],
        ]
        self.discrete = torch.cat(discrete, -1).long()

        perm = torch.rand(B, S * A).argsort(dim=1)

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
