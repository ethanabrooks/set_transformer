from typing import Optional

import torch
import torch.nn.functional as F

from models.set_transformer import SetTransformer as Base
from utils import DataPoint


class SetTransformer(Base):
    def forward(
        self, x: DataPoint, values: torch.Tensor, q_values: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        discrete = torch.stack(
            [
                x.states,
                x.actions,
                x.next_states,
                x.rewards,
            ],
            dim=-1,
        )
        discrete: torch.Tensor = self.embedding(discrete.long())
        _, _, _, D = discrete.shape
        continuous = torch.cat([x.action_probs, values[..., None]], dim=-1)
        continuous = self.positional_encoding.forward(continuous)
        X = torch.cat([continuous, discrete], dim=-2)
        B, S, T, D = X.shape
        X = X.reshape(B * S, T, D)
        _, _, D = X.shape
        assert [*X.shape] == [B * S, T, D]
        Y: torch.Tensor = self.transition_encoder(X)
        assert [*Y.shape] == [B * S, T, D]
        Y = Y.reshape(B, S, T, D).sum(2)
        assert [*Y.shape] == [B, S, D]
        Z: torch.Tensor = self.sequence_network(Y)
        assert [*Z.shape] == [B, S, D]
        outputs: torch.Tensor = self.dec(Z)

        loss = F.mse_loss(outputs, q_values.float())
        return outputs, loss
