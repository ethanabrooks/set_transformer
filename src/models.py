import math

import torch
import torch.nn as nn
from torch import Tensor

from modules import ISAB


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

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


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
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = src * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output


class GRU(nn.Module):
    def __init__(self, dim_hidden):
        super().__init__()
        self.gru = nn.GRU(dim_hidden, dim_hidden, batch_first=False)

    def forward(self, x):
        h, _ = self.gru(x)
        return h


class SetTransformer(nn.Module):
    def __init__(
        self,
        ntoken,
        dim_hidden=128,
        ln=False,
        num_inds=32,
        num_heads=4,
        seq2seq="gru",
    ):
        super(SetTransformer, self).__init__()
        self.embedding = nn.Embedding(ntoken, dim_hidden)
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
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        # PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        self.dec = nn.Linear(dim_hidden, ntoken)

    def forward(self, X):
        B, T, S = X.shape
        X = X.reshape(B * T, S).T
        X = self.embedding(X)
        _, _, D = X.shape
        assert [*X.shape] == [S, B * T, D]
        Y = self.seq2seq(X)
        assert [*Y.shape] == [S, B * T, D]
        Y = Y.reshape(S, B, T, D).sum(2)
        assert [*Y.shape] == [S, B, D]
        Y = Y.swapaxes(0, 1)
        assert [*Y.shape] == [B, S, D]
        Y = self.enc(Y)
        assert [*Y.shape] == [B, S, D]
        return self.dec(Y)
