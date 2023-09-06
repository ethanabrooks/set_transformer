import torch
from torch.utils.data import Dataset


class RLData(Dataset):
    def __init__(self, num_steps, seq_len):
        states = torch.randint(0, int(10), (num_steps, 2, seq_len))
        goals = torch.randint(0, int(10), (num_steps, 2, 1))
        actions = torch.randint(0, 4, (num_steps, seq_len))
        mapping = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]])
        delta = mapping[actions].swapaxes(1, 2)

        rewards = (states - goals).sum(1).abs().float()
        slow = 1 + torch.rand(rewards.shape, device=rewards.device)
        rewards *= slow
        rewards = rewards.round().long()

        Z = (states + delta - goals).sum(1).abs().float()
        Z *= slow
        Z = Z.round().long()

        self.X = (
            torch.cat([states, actions[:, None], rewards[:, None]], 1).long().cuda()
        )
        self.Z = Z.cuda()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Z[idx]
