import torch
from torch.utils.data import Dataset


class RLData(Dataset):
    def __init__(self, grid_size, n_steps, seq_len):
        states = torch.randint(0, grid_size, (n_steps, seq_len, 2))
        goals = torch.randint(0, grid_size, (n_steps, 1, 2)).expand_as(states)
        actions = torch.randint(0, 4, (n_steps, seq_len))
        mapping = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
        deltas = mapping[actions]
        order = torch.randint(0, 2 * grid_size, (n_steps, seq_len))
        rewards = (states == goals).all(-1).long()
        Q = (states + deltas - goals).sum(-1).abs()
        Q_ = torch.min(Q, order)
        self.X = (
            torch.cat(
                [states, actions[..., None], rewards[..., None], Q_[..., None]], -1
            )
            .long()
            .cuda()
        )

        self.Z = torch.min(Q, order + 1).cuda()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx]
