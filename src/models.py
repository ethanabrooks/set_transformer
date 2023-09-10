import math

import torch
import torch.nn as nn
from torch import Tensor

from modules import ISAB, SAB


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:  # dead: disable
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def _ff_block(self, x: Tensor) -> Tensor:  # dead: disable
        return x


class TransformerModel(nn.Module):
    # https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:  # dead: disable
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, d_model]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]``
        """
        src = src.swapaxes(0, 1)
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output.swapaxes(0, 1)


class GRU(nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()
        self.gru = nn.GRU(dim_hidden, dim_hidden, batch_first=True)

    def forward(self, x):  # dead: disable
        h, _ = self.gru(x)
        return h


class SetTransformer(nn.Module):
    def __init__(
        self,
        n_tokens,
        dim_output,
        n_isab,
        n_sab,
        dim_hidden=128,
        ln=False,
        num_inds=32,
        num_heads=4,
        seq2seq="gru",
    ):
        super(SetTransformer, self).__init__()
        self.embedding = nn.Embedding(n_tokens, dim_hidden)
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        if seq2seq == "transformer":
            self.seq2seq = TransformerModel(
                d_model=dim_hidden,
                nhead=num_heads,
                d_hid=dim_hidden,
                nlayers=1,
            )
        elif seq2seq == "gru":
            self.seq2seq = GRU(dim_hidden)
        else:
            raise ValueError(f"Unknown seq2seq {seq2seq}")

        self.enc = nn.Sequential(
            *[
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
                for _ in range(n_isab)
            ],
            *[SAB(dim_hidden, dim_hidden, num_heads, ln=ln) for _ in range(n_sab)],
        )
        # PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        self.dec = nn.Linear(dim_hidden, dim_output)

    def forward(self, X):  # dead: disable
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
