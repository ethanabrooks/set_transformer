import math

import torch
from torch.utils.data import Dataset


class RLData(Dataset):
    def __init__(self, n_token, n_steps, seq_len):
        states = torch.randint(0, int(math.sqrt(n_token)), (n_steps, seq_len, 2))
        goals = torch.randint(0, int(math.sqrt(n_token)), (n_steps, 1, 2)).expand_as(
            states
        )
        rewards = (states == goals).all(-1).long()
        actions = torch.randint(0, 4, (n_steps, seq_len))
        mapping = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
        deltas = mapping[actions]
        self.X = (
            torch.cat([states, rewards[..., None], actions[..., None]], -1)
            .long()
            .cuda()
        )
        Z = (states + deltas - goals).sum(-1).abs().cuda()

        self.Z = Z.cuda()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx]
