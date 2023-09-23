import torch
from torch.utils.data import Dataset

from utils import policy_evaluation, round_tensor


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
        Pi, R, V = policy_evaluation(
            grid_size=grid_size,
            n_policies=n_policies,
            n_rounds=n_rounds,
            n_steps=n_steps,
        )
        mapping = torch.tensor([-1, 1])
        A = len(mapping)  # number of actions
        all_states = torch.arange(grid_size)
        seq_len = A * len(all_states)

        def get_indices(states: torch.Tensor):
            # In 1D, the state itself is the index
            return torch.arange(n_steps)[:, None], states

        n_rounds = 2 * grid_size
        Pi, R, V = policy_evaluation(
            grid_size=grid_size,
            n_policies=n_policies,
            n_rounds=n_rounds,
            n_steps=n_steps,
        )

        states = all_states[None].tile(n_steps, A, 1).reshape(n_steps, -1)

        # In the 1D case, states are already the indices
        S = states

        # Gather the corresponding probabilities from Pi
        probabilities = Pi.gather(1, S[..., None]).expand(-1, -1, A)

        print("Sampling actions...", end="", flush=True)
        actions = (
            torch.arange(A)[:, None]
            .expand(-1, len(all_states))
            .reshape(-1)
            .expand(n_steps, -1)
        )
        print("✓")

        deltas = mapping[actions]
        next_states = torch.clamp(states + deltas, 0, grid_size - 1)

        idxs1, idxs2 = get_indices(states)
        rewards = R[idxs1, idxs2].gather(dim=2, index=actions[..., None])

        if max_order is None:
            max_order = len(V) - 2
        if min_order is None:
            min_order = 0

        order = torch.randint(min_order, max_order + 1, (n_steps, 1)).tile(1, seq_len)
        idxs1, idxs2 = get_indices(next_states)

        V1 = V[order, idxs1, idxs2]
        V2 = V[order + 1, idxs1, idxs2]

        print("Computing unique values...", end="", flush=True)
        V1 = round_tensor(V1, n_input_bins)
        print("✓", end="", flush=True)
        V2 = round_tensor(V2, n_output_bins)
        probabilities = round_tensor(probabilities, n_input_bins)
        print("✓")

        self.X = (
            torch.cat(
                [
                    states[..., None],
                    probabilities,
                    actions[..., None],
                    next_states[..., None],
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
