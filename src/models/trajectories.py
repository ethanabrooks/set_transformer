from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from transformers import GPT2Config, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from models.set_transformer import ISAB, SAB
from models.set_transformer import SetTransformer as Base
from utils import DataPoint


def get_input_bellman(n_bellman: int, bellman_delta: int):
    return 1 + n_bellman - bellman_delta


class Model(Base):
    def __init__(
        self,
        bellman_delta: int,
        n_actions: int,
        n_hidden: int,
        n_rotations: int,
        n_tokens: int,
        positional_encoding_args: dict,
        **transformer_args: dict
    ):
        super().__init__(
            n_actions, n_hidden, n_tokens, positional_encoding_args, **transformer_args
        )
        self.bellman_delta = bellman_delta
        if bellman_delta > 1:
            self.input_bellman_embedding = nn.Embedding(bellman_delta - 1, n_hidden)
        self.n_rotations = n_rotations

    def forward(self, x: DataPoint) -> tuple[torch.Tensor, torch.Tensor]:
        action_probs: torch.Tensor = self.offset(x.action_probs)
        actions: torch.Tensor = self.offset(x.actions)
        done: torch.Tensor = self.offset(x.done)
        next_obs: torch.Tensor = self.offset(x.next_obs)
        rewards: torch.Tensor = self.offset(x.rewards)
        discrete = torch.stack(
            [
                x.obs,
                actions,
                done,
                next_obs,
                rewards,
            ],
            dim=-1,
        )
        discrete: torch.Tensor = self.embedding(discrete.long())
        _, _, _, D = discrete.shape
        values = (x.input_q * x.action_probs).sum(-1)
        values = self.positional_encoding(values)
        if self.bellman_delta > 1:
            input_bellman = get_input_bellman(
                x.n_bellman, self.bellman_delta
            )  # Bellman number of input values
            use_embedding = (input_bellman < 0)[
                :, None, None
            ]  # when input Bellman are negative, values are clamped/meaningless
            input_bellman = torch.clamp(
                input_bellman, 0, self.input_bellman_embedding.num_embeddings - 1
            )  # clamp embeddings for positive input Bellman
            input_bellman_embedding = self.input_bellman_embedding(input_bellman)[
                :, None
            ]
            values = (
                input_bellman_embedding * use_embedding + values * ~use_embedding
            )  # use embedding when input Bellman is negative
        action_probs = self.positional_encoding(action_probs)
        continuous = torch.cat([action_probs, values[:, :, None]], dim=-2)
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
        if x.target_q is None:
            loss = None
        else:
            assert [*x.target_q.shape] == [B, S]

            loss = F.mse_loss(
                outputs[torch.arange(B)[:, None], torch.arange(S)[None], x.actions],
                x.target_q,
            )
        return outputs, loss

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

    def forward_with_rotation(
        self, x: DataPoint, optimizer: Optional[Optimizer]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_orig = x
        agg_loss = 0
        agg_outputs = None
        updated = None
        _, l = x.rewards.shape
        rotation_unit = l // self.n_rotations
        for rotation_index in range(self.n_rotations):
            rotation_shift = rotation_index * rotation_unit

            def rotate(x: torch.Tensor):
                if x is None or x.ndim == 1:
                    return x
                return torch.roll(x, shifts=rotation_shift, dims=1)

            x_cpu = DataPoint(*[rotate(x) for x in x_orig])
            x = DataPoint(*[x if x is None else x.cuda() for x in x_cpu])
            rng_rot = torch.roll(torch.arange(l), shifts=rotation_shift)
            if optimizer is not None:
                optimizer.zero_grad()
            outputs: torch.Tensor
            loss: torch.Tensor
            outputs, loss = self.forward(x=x)
            if loss is not None:
                agg_loss += loss

            tail_idxs = torch.arange(l - rotation_unit, l)
            if agg_outputs is None:
                assert updated is None
                agg_outputs = torch.zeros_like(outputs)
                updated = torch.zeros_like(outputs)
            agg_outputs[:, rng_rot[tail_idxs]] = outputs[:, tail_idxs]
            updated[:, rng_rot[tail_idxs]] = 1

            if optimizer is not None:
                loss.backward()
                optimizer.step()
        return agg_outputs, agg_loss


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
        embedded_obs: torch.Tensor = self.embedding(x.obs)
        assert [*embedded_obs.shape] == [B, S, D]
        Y = torch.cat([embedded_discrete, embedded_obs], dim=1)
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
