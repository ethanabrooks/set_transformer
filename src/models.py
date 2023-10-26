from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ISAB, SAB


class GRU(nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()
        self.gru = nn.GRU(dim_hidden, dim_hidden, batch_first=True)

    def forward(self, x):
        h, _ = self.gru(x)
        return h


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        log_term: float,
        n_hidden: int,
        scale_term: float,
    ):
        super().__init__()
        self.n_hidden = n_hidden
        div_term = scale_term * torch.exp(
            torch.arange(0, self.n_hidden, 2)
            * -(torch.log(torch.tensor(log_term)) / self.n_hidden)
        )
        self.register_buffer("div_term", div_term)

    @lru_cache()
    def encoding(self, shape: torch.Size, device: torch.device):
        return torch.zeros(*shape[:-1], self.n_hidden).to(device)

    def forward(self, x: torch.Tensor):
        """
        Encode continuous values using sinusoidal functions.

        Args:
        - continuous (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the continuous values.
        - d_model (int): Dimension of the encoding. Typically the model's hidden dimension.

        Returns:
        - torch.Tensor: The sinusoidal encoding of shape (batch_size, sequence_length, d_model).
        """
        # Expand dimensions for broadcasting
        x = x.unsqueeze(-1)
        pos = x * self.div_term
        encoding = self.encoding(x.shape, x.device)
        encoding[..., 0::2] = torch.sin(pos)
        encoding[..., 1::2] = torch.cos(pos)

        return encoding


class SetTransformer(nn.Module):
    def __init__(
        self,
        isab_args: dict,
        n_actions: int,
        n_isab: int,
        n_hidden: int,
        n_sab: int,
        n_tokens: int,
        positional_encoding_args: dict,
        sab_args: dict,
    ):
        super(SetTransformer, self).__init__()
        self.embedding = nn.Embedding(n_tokens, n_hidden)
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.n_hidden = n_hidden
        self.positional_encoding = PositionalEncoding(
            n_hidden=n_hidden, **positional_encoding_args
        )
        self.seq2seq = GRU(n_hidden)

        self.network = nn.Sequential(
            *[ISAB(n_hidden, n_hidden, **isab_args, **sab_args) for _ in range(n_isab)],
            *[SAB(n_hidden, n_hidden, **sab_args) for _ in range(n_sab)],
        )
        # PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        self.dec = nn.Linear(n_hidden, n_actions)

    def forward(
        self,
        v1: torch.Tensor,
        action_probs: torch.Tensor,
        discrete: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        discrete = self.embedding(discrete)
        _, _, _, D = discrete.shape
        continuous = torch.cat([action_probs, v1[..., None]], dim=-1)
        continuous = self.positional_encoding.forward(continuous)
        X = torch.cat([continuous, discrete], dim=-2)
        B, S, T, D = X.shape
        X = X.reshape(B * S, T, D)
        _, _, D = X.shape
        assert [*X.shape] == [B * S, T, D]
        Y: torch.Tensor = self.seq2seq(X)
        assert [*Y.shape] == [B * S, T, D]
        Y = Y.reshape(B, S, T, D).sum(2)
        assert [*Y.shape] == [B, S, D]
        Z: torch.Tensor = self.network(Y)
        assert [*Z.shape] == [B, S, D]
        outputs: torch.Tensor = self.dec(Z)

        loss: torch.Tensor = F.mse_loss(outputs, targets.float())
        return outputs, loss
