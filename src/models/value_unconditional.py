from typing import NamedTuple

import torch
import torch.nn.functional as F

from models.set_transformer import SetTransformer as BaseSetTransformer


class DataPoint(NamedTuple):
    action_probs: torch.Tensor
    actions: torch.Tensor
    idx: torch.Tensor
    n_bellman: torch.Tensor
    next_states: torch.Tensor
    q_values: torch.Tensor
    rewards: torch.Tensor
    states: torch.Tensor


class SetTransformer(BaseSetTransformer):
    def forward(
        self, x: DataPoint, q_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        discrete = torch.stack(
            [
                x.states,
                x.actions,
                x.next_states,
                x.rewards,
                x.n_bellman[:, None].expand_as(x.rewards),
            ],
            dim=-1,
        )
        discrete: torch.Tensor = self.embedding(discrete.long())
        _, _, _, D = discrete.shape
        continuous = self.positional_encoding.forward(x.action_probs)
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

        loss = F.mse_loss(outputs, q_values.float())
        return outputs, loss
