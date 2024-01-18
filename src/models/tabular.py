from typing import Optional

import torch
import torch.nn.functional as F

from models.set_transformer import SetTransformer as Base
from utils.dataclasses import DataPoint


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
        _, _, _, d = discrete.shape
        values = (x.input_q * x.action_probs).sum(-1, keepdim=True)
        continuous = torch.cat([x.action_probs, values], dim=-1)
        continuous = self.positional_encoding.forward(continuous)
        X = torch.cat([continuous, discrete], dim=-2)
        b, s, t, d = X.shape
        X = X.reshape(b * s, t, d)
        _, _, d = X.shape
        assert [*X.shape] == [b * s, t, d]
        Y: torch.Tensor = self.transition_encoder(X)
        assert [*Y.shape] == [b * s, t, d]
        Y = Y.reshape(b, s, t, d).sum(2)
        assert [*Y.shape] == [b, s, d]
        Z: torch.Tensor = self.sequence_network(Y)
        assert [*Z.shape] == [b, s, d]
        outputs: torch.Tensor = self.dec(Z)

        if x.target_q is None:
            return outputs, None
        loss = F.mse_loss(outputs, x.target_q.float())
        return outputs, loss
