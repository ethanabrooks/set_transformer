import math

import torch
import torch.nn as nn
from torch import Tensor

from modules import ISAB


class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output),
        )

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X


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
        ntoken: int,
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
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, d_model]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]``
        """
        src = src.swapaxes(0, 1)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        return output.swapaxes(0, 1)


class SetTransformer(nn.Module):
    def __init__(
        self,
        ntoken,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.transformer = TransformerModel(
            ntoken=ntoken,
            d_model=dim_hidden,
            nhead=num_heads,
            d_hid=dim_hidden,
            nlayers=1,
        )
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
        B, S, T = X.shape
        X = X.reshape(B * S, T)
        Y = self.transformer(X)
        _, _, D = Y.shape
        assert [*Y.shape] == [B * S, T, D]
        Y = Y.reshape(B, S, T, D).sum(2)
        assert [*Y.shape] == [B, S, D]
        Y = self.enc(Y)
        assert [*Y.shape] == [B, S, D]
        return self.dec(Y)
