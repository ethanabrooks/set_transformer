import torch
import torch.nn as nn
import torch.nn.functional as F

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
        continuous_max: torch.Tensor,
        continuous_min: torch.Tensor,
        isab_args: dict,
        loss_type: LossType,
        n_isab: int,
        n_hidden: int,
        n_output: int,
        n_sab: int,
        n_tokens: int,
        sab_args: dict,
    ):
        super(SetTransformer, self).__init__()
        self.continuous_max = continuous_max
        self.continuous_min = continuous_min
        self.embedding = nn.Embedding(n_tokens, n_hidden)
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.loss_type = loss_type
        self.n_hidden = n_hidden
        self.seq2seq = GRU(n_hidden)

        self.network = nn.Sequential(
            *[ISAB(n_hidden, n_hidden, **isab_args, **sab_args) for _ in range(n_isab)],
            *[SAB(n_hidden, n_hidden, **sab_args) for _ in range(n_sab)],
        )
        # PMA(dim_hidden, num_heads, num_outputs, ln=ln),
        # SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
        if loss_type == LossType.MSE:
            n_output = 1
        self.dec = nn.Linear(n_hidden, n_output)

    def sinusoidal_encoding(self, continuous: torch.Tensor):
        """
        Encode continuous values using sinusoidal functions.

        Args:
        - continuous (torch.Tensor): A tensor of shape (batch_size, sequence_length) containing the continuous values.
        - d_model (int): Dimension of the encoding. Typically the model's hidden dimension.

        Returns:
        - torch.Tensor: The sinusoidal encoding of shape (batch_size, sequence_length, d_model).
        """
        # Expand dimensions for broadcasting
        continuous = continuous.unsqueeze(-1)
        div_term = torch.exp(
            torch.arange(0, self.n_hidden, 2)
            * -(torch.log(torch.tensor(10000.0)) / self.n_hidden)
        )
        div_term = div_term.to(continuous.device)

        pos = continuous * div_term
        encoding = torch.zeros(*continuous.shape[:-1], self.n_hidden).to(
            continuous.device
        )
        encoding[..., 0::2] = torch.sin(pos)
        encoding[..., 1::2] = torch.cos(pos)

        return encoding

    def forward(
        self, continuous: torch.Tensor, discrete: torch.Tensor, targets: torch.Tensor
    ):
        discrete = self.embedding(discrete)
        _, _, _, D = discrete.shape
        continuous = self.sinusoidal_encoding(continuous)
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

        if self.loss_type == LossType.MSE:
            loss: torch.Tensor = F.mse_loss(outputs.squeeze(-1), targets.float())
        elif self.loss_type == LossType.CROSS_ENTROPY:
            loss: torch.Tensor = F.cross_entropy(outputs.swapaxes(1, 2), targets)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        return outputs, loss
