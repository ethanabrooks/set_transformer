import random
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from transformers import GPT2Config, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from models.set_transformer import SetTransformer as Base
from ppo.networks import Flatten
from utils import DataPoint


def get_input_bellman(n_bellman: int, bellman_delta: int):
    return 1 + n_bellman - bellman_delta


class GPT2(nn.Module):
    def __init__(
        self, n_ctx: int, n_heads: int, n_hidden: int, n_layers: int, **kwargs
    ):
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


class Model(Base, ABC):
    def __init__(
        self,
        bellman_delta: int,
        n_actions: int,
        n_hidden: int,
        n_rotations: int,
        n_tokens: int,
        pad_value: int,
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
        self.obs_encoder = self.build_obs_encoder()
        self.rew_encoder = self.build_rew_encoder()
        self.pad_value = pad_value

    @abstractmethod
    def build_obs_encoder(self, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def build_rew_encoder(self, **kwargs) -> nn.Module:
        pass

    def build_sequence_network(self, **kwargs):
        return GPT2(**kwargs)

    def forward(
        self, x: DataPoint, unmasked_actions: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if unmasked_actions is None:
            unmasked_actions = x.actions
        action_probs: torch.Tensor = self.offset(x.action_probs)
        actions: torch.Tensor = self.offset(x.actions)
        done: torch.Tensor = self.offset(x.done)
        next_obs: torch.Tensor = self.offset(x.next_obs)
        rewards: torch.Tensor = self.offset(x.rewards)
        b, l, *o = x.obs.shape
        obs: torch.Tensor = self.obs_encoder(x.obs.reshape(b * l, *o))
        actions: torch.Tensor = self.embedding(actions.long())
        done: torch.Tensor = self.embedding(done.long())
        next_obs: torch.Tensor = self.obs_encoder(next_obs.reshape(b * l, *o))
        rewards: torch.Tensor = self.rew_encoder(rewards.long())
        discrete = torch.stack(
            [
                obs.reshape(b, l, self.n_hidden),
                actions,
                done,
                next_obs.reshape(b, l, self.n_hidden),
                rewards,
            ],
            dim=-2,
        )
        d = self.n_hidden
        t = 5
        assert [*discrete.shape] == [b, l, t, d]
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
        b, l, t, d = X.shape
        X = X.reshape(b * l, t, d)
        _, _, d = X.shape
        assert [*X.shape] == [b * l, t, d]
        Y: torch.Tensor = self.transition_encoder(X)
        assert [*Y.shape] == [b * l, t, d]
        embedded_discrete = Y.reshape(b, l, t, d).sum(2)
        assert [*embedded_discrete.shape] == [b, l, d]
        outputs: torch.Tensor = self.forward_output(
            embedded_discrete=embedded_discrete, x=x
        )
        assert [*outputs.shape][:-1] == [b, l]
        if x.target_q is None:
            loss = None
        else:
            assert [*x.target_q.shape] == [b, l]

            loss = F.mse_loss(
                outputs[
                    torch.arange(b)[:, None], torch.arange(l)[None], unmasked_actions
                ],
                x.target_q,
            )
        return outputs, loss

    def forward_output(
        self, embedded_discrete: torch.Tensor, x: DataPoint
    ) -> torch.Tensor:
        b, s, d = embedded_discrete.shape
        Z: torch.Tensor = self.sequence_network(embedded_discrete)
        assert [*Z.shape] == [b, s, d]
        outputs: torch.Tensor = self.dec(Z)
        assert [*outputs.shape][:-1] == [b, s]
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
        x_cuda = DataPoint(*[y if y is None else y.cuda() for y in x])
        x_mask = deepcopy(x_cuda)
        x_mask.action_probs[:, -1] = self.pad_value
        x_mask.actions[:, -1] = self.pad_value
        x_mask.done[:, -1] = self.pad_value
        x_mask.next_obs[:, -1] = self.pad_value
        x_mask.rewards[:, -1] = self.pad_value

        agg_loss = 0
        agg_outputs = None
        updated = None
        _, l = x.rewards.shape
        rotation_unit = l // self.n_rotations
        rotation_start = random.randint(
            0, l - 1
        )  # randomize rotation to break correlation with termination at end of sequence
        for rotation_index in range(self.n_rotations):
            rotation_shift = rotation_start + rotation_index * rotation_unit

            def rotate(x: torch.Tensor):
                if x is None or x.ndim == 1:
                    return x
                return torch.roll(x, shifts=rotation_shift, dims=1)

            x = DataPoint(*[rotate(x) for x in x_mask])
            unmasked_actions = rotate(x_cuda.actions)
            rng_rot = torch.roll(torch.arange(l), shifts=rotation_shift)
            if optimizer is not None:
                optimizer.zero_grad()
            outputs: torch.Tensor
            loss: torch.Tensor
            outputs, loss = self.forward(x=x, unmasked_actions=unmasked_actions)
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
        assert torch.all(updated)
        return agg_outputs, agg_loss


class NormalizeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x / 255.0


class MiniWorldModel(Model):
    def build_obs_encoder(self):
        return nn.Sequential(
            NormalizeLayer(),
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 4 * 6, self.n_hidden),
            nn.ReLU(),
        )

    def build_rew_encoder(self):
        return self.positional_encoding


class CastToLongLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.long()


class GridWorldModel(Model):
    def build_obs_encoder(self):
        return nn.Sequential(CastToLongLayer(), self.embedding)

    def build_rew_encoder(self):
        return nn.Sequential(CastToLongLayer(), self.embedding)
