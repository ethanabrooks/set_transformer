from typing import Optional

import torch
import torch.nn.functional as F

from models.set_transformer import SetTransformer as Base
from utils import DataPoint


class SetTransformer(Base):
    def forward(self, x: DataPoint) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        discrete = torch.stack(
            [
                x.obs,
                x.actions,
                x.next_obs,
                x.rewards,
            ],
            dim=-1,
        )
        discrete: torch.Tensor = self.embedding(discrete.long())
        _, _, _, D = discrete.shape
        values = (x.input_q * x.action_probs).sum(-1, keepdim=True)
        continuous = torch.cat([x.action_probs, values], dim=-1)
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

        if x.target_q is None:
            return outputs, None
        loss = F.mse_loss(outputs, x.target_q.float())
        return outputs, loss
