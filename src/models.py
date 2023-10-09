import torch.nn as nn

from metrics import LossType
from modules import ISAB, SAB


class GRU(nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()
        self.gru = nn.GRU(dim_hidden, dim_hidden, batch_first=True)

    def forward(self, x):
        h, _ = self.gru(x)
        return h


class SetTransformer(nn.Module):
    def __init__(
        self,
        n_tokens,
        dim_output,
        loss_type: LossType,
        n_isab,
        n_sab,
        dim_hidden=128,
        ln=False,
        num_inds=32,
        num_heads=8,
    ):
        super(SetTransformer, self).__init__()
        self.embedding = nn.Embedding(n_tokens, dim_hidden)
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.seq2seq = GRU(dim_hidden)

        self.enc = nn.Sequential(
            *[
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
                for _ in range(n_isab)
            ],
            *[SAB(dim_hidden, dim_hidden, num_heads, ln=ln) for _ in range(n_sab)],
        )
        # PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        if loss_type == LossType.MSE:
            dim_output = 1
        self.dec = nn.Linear(dim_hidden, dim_output)

    def forward(self, X):
        B, S, T = X.shape
        X = X.reshape(B * S, T)
        X = self.embedding(X)
        _, _, D = X.shape
        assert [*X.shape] == [B * S, T, D]
        Y = self.seq2seq(X)
        assert [*Y.shape] == [B * S, T, D]
        Y = Y.reshape(B, S, T, D).sum(2)
        assert [*Y.shape] == [B, S, D]
        Y = self.enc(Y)
        assert [*Y.shape] == [B, S, D]
        return self.dec(Y)
