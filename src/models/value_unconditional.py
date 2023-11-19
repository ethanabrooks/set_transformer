from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from models.set_transformer import ISAB, SAB
from models.set_transformer import SetTransformer as Base


class DataPoint(NamedTuple):
    action_probs: torch.Tensor
    actions: torch.Tensor
    idx: torch.Tensor
    n_bellman: torch.Tensor
    next_states: torch.Tensor
    q_values: torch.Tensor
    rewards: torch.Tensor
    states: torch.Tensor


class Model(Base):
    def forward(
        self, x: DataPoint, q_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_probs: torch.Tensor = self.offset(x.action_probs)
        actions: torch.Tensor = self.offset(x.actions)
        next_states: torch.Tensor = self.offset(x.next_states)
        rewards: torch.Tensor = self.offset(x.rewards)
        discrete = torch.stack(
            [
                x.states,
                actions,
                next_states,
                rewards,
                x.n_bellman[:, None].expand_as(rewards),
            ],
            dim=-1,
        )
        discrete: torch.Tensor = self.embedding(discrete.long())
        _, _, _, D = discrete.shape
        continuous = self.positional_encoding.forward(action_probs)
        X = torch.cat([continuous, discrete], dim=-2)
        B, S, T, D = X.shape
        X = X.reshape(B * S, T, D)
        _, _, D = X.shape
        assert [*X.shape] == [B * S, T, D]
        Y: torch.Tensor = self.transition_encoder(X)
        assert [*Y.shape] == [B * S, T, D]
        embedded_discrete = Y.reshape(B, S, T, D).sum(2)
        assert [*embedded_discrete.shape] == [B, S, D]
        outputs: torch.Tensor = self.forward_output(
            embedded_discrete=embedded_discrete, x=x
        )
        assert [*outputs.shape][:-1] == [B, S]
        assert [*q_values.shape] == [B, S]

        loss = F.mse_loss(
            outputs[torch.arange(B)[:, None], torch.arange(S)[None], x.actions],
            q_values,
            reduction="none",
        )
        return outputs, loss.mean()

    def forward_output(
        self, embedded_discrete: torch.Tensor, x: DataPoint
    ) -> torch.Tensor:
        B, S, D = embedded_discrete.shape
        Z: torch.Tensor = self.sequence_network(embedded_discrete)
        assert [*Z.shape] == [B, S, D]
        outputs: torch.Tensor = self.dec(Z)
        assert [*outputs.shape][:-1] == [B, S]
        return outputs

    def offset(self, x: torch.Tensor) -> torch.Tensor:
        def padding():
            padding = [1, 0]
            for _ in range(len(x.shape) - 2):
                padding = [0, 0] + padding
            return padding

        return F.pad(x[:, :-1], tuple(padding()), value=self.n_tokens)


class CausalTransformer(Model):
    def build_sequence_network(
        self, n_ctx: int, n_heads: int, n_hidden: int, n_layers: int, **kwargs
    ):
        class GPT2(nn.Module):
            def __init__(self):
                super().__init__()
                # Deal with the insanity of the GPT2 API
                config = GPT2Config(
                    vocab_size=1,  # dummy
                    n_layer=n_layers,
                    n_layers=n_layers,
                    num_hidden_layers=n_layers,
                    n_embd=n_hidden,
                    n_positions=n_ctx,
                    n_heads=n_heads,
                    num_attention_heads=n_heads,
                    num_heads=n_heads,
                    **kwargs,
                )
                self.gpt2 = GPT2Model(config)

            def forward(self, x: torch.Tensor):
                hidden_states: BaseModelOutputWithPastAndCrossAttentions = self.gpt2(
                    inputs_embeds=x
                )
                return hidden_states.last_hidden_state

        return GPT2()


class GRU(Model):
    def build_sequence_network(self, n_hidden: int, n_layers: int):
        class GRU(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.gru = nn.GRU(
                    input_size=n_hidden,
                    hidden_size=n_hidden,
                    batch_first=True,
                    num_layers=n_layers,
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                full, _ = self.gru(x)  # _ == last
                return full

        return GRU()


class SetTransformer(Model):
    def build_sequence_network(
        self,
        isab_args: dict,
        n_hidden: int,
        n_isab: int,
        n_sab: int,
        sab_args: dict,
    ):
        return nn.Sequential(
            *[ISAB(n_hidden, n_hidden, **isab_args, **sab_args) for _ in range(n_isab)],
            *[SAB(n_hidden, n_hidden, **sab_args) for _ in range(n_sab)],
        )

    def forward_output(
        self, embedded_discrete: torch.Tensor, x: DataPoint
    ) -> torch.Tensor:
        B, S, D = embedded_discrete.shape
        embedded_states: torch.Tensor = self.embedding(x.states)
        assert [*embedded_states.shape] == [B, S, D]
        Y = torch.cat([embedded_discrete, embedded_states], dim=1)
        Z: torch.Tensor
        Z = self.sequence_network(Y)
        assert [*Z.shape] == [B, 2 * S, D]
        outputs: torch.Tensor = self.dec(Z)
        assert [*outputs.shape][:-1] == [B, 2 * S]
        outputs = outputs[:, S:]
        return outputs

    @staticmethod
    def offset(x: torch.Tensor) -> torch.Tensor:
        return x
