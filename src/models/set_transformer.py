import math
from functools import lru_cache
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataPoint(NamedTuple):
    idx: torch.Tensor
    input_bellman: torch.Tensor
    continuous: torch.Tensor
    discrete: torch.Tensor
    q_values: torch.Tensor
    values: torch.Tensor


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, n_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, n_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, n_heads, n_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, n_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, n_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, n_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


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
        continuous: torch.Tensor,
        discrete: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        discrete = self.embedding(discrete)
        _, _, _, D = discrete.shape
        continuous = torch.cat([continuous, v1[..., None]], dim=-1)
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

        loss = None if targets is None else F.mse_loss(outputs, targets.float())
        return outputs, loss
