import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box

from ppo.envs.envs import Sequence
from ppo.utils import init


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), -1)


class Network(nn.Module):
    def __init__(
        self,
        recurrent: bool,
        recurrent_input_size: int,
        hidden_size: int,
        num_tasks: Optional[int] = None,
    ):
        super(Network, self).__init__()
        self.task_embedding = (
            None if num_tasks is None else nn.Embedding(num_tasks, hidden_size)
        )
        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if "bias" in name:
                    nn.init.constant_(param, 0)
                elif "weight" in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x: torch.Tensor, hxs: torch.Tensor, masks: torch.Tensor):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x[None], (hxs * masks[:, None])[None])
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros: torch.Tensor = (
                (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
            )

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs[None]
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx], hxs * masks[start_idx].view(1, -1, 1)
                )

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


def init_conv(module: nn.Module):
    return init(
        module,
        nn.init.orthogonal_,
        lambda x: nn.init.constant_(x, 0),
        nn.init.calculate_gain("relu"),
    )


class CNNBase(Network):
    def __init__(self, hidden_size: int, num_inputs: int, **kwargs):
        super(CNNBase, self).__init__(
            recurrent_input_size=hidden_size, hidden_size=hidden_size, **kwargs
        )

        self.main = nn.Sequential(
            init_conv(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_conv(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_conv(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_conv(nn.Linear(32 * 4 * 6, hidden_size)),
            nn.ReLU(),
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def embed_inputs(self, inputs: torch.Tensor):
        return self.main(inputs / 255.0)

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        rnn_hxs: torch.Tensor,
        tasks: torch.Tensor = None,
    ):
        x = self.embed_inputs(inputs)
        if tasks is not None and self.task_embedding is not None:
            x = x + self.task_embedding(tasks)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        values = self.critic_linear.forward(x).squeeze(-1)
        return values, x, rnn_hxs


class SequenceEnvEmbedding(nn.Module):
    def __init__(self, n_objects: int, n_hidden: int) -> None:
        super().__init__()
        self.obj_embedding = nn.Embedding(n_objects, n_hidden)
        self.seq_embedding = nn.GRU(n_hidden, n_hidden, batch_first=True)

    def forward(self, x: torch.Tensor):
        x = self.obj_embedding(x.long())
        _, [x] = self.seq_embedding(x)
        return x


RGB_SIZE = math.prod(Sequence.miniworld_obs_shape)


class NormalizeLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x / 255.0


class MiniWorldObEncoder(nn.Module):
    def __init__(self, n_hidden: int, n_tokens: int, obs_space: Box) -> None:
        super().__init__()
        self.main = nn.Sequential(
            NormalizeLayer(),
            init_conv(nn.Conv2d(3, 32, 8, stride=4)),
            nn.ReLU(),
            init_conv(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_conv(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_conv(nn.Linear(32 * 4 * 6, n_hidden)),
            nn.ReLU(),
        )
        self.obs_space = obs_space
        if len(obs_space.shape) == 1:
            high = obs_space.high
            sequence_high = high[RGB_SIZE:]
            n_objects = int(sequence_high.max())
            self.sequence_env_embedding = SequenceEnvEmbedding(
                1 + max(n_objects, n_tokens), n_hidden
            )
        elif obs_space.shape == Sequence.miniworld_obs_shape:
            self.sequence_env_embedding = None
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor):
        _, *o = x.shape
        if self.sequence_env_embedding is None:
            assert o == Sequence.miniworld_obs_shape
            return self.main(x)
        else:
            assert len(o) == 1
            n_sequence = x.size(-1) - RGB_SIZE
            rgb, sequence = torch.split(x, [RGB_SIZE, n_sequence], dim=-1)
            rgb = (
                rgb.reshape(-1, *Sequence.miniworld_obs_shape)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            rgb = self.main(rgb)
            sequence = self.sequence_env_embedding(sequence)
            return rgb + sequence
        raise NotImplementedError


class SequenceBase(CNNBase):
    def __init__(
        self,
        hidden_size: int,
        n_objects: int,
        n_permutations: int,
        n_sequence: int,
        num_inputs: int,
        permutation_starting_idx: int,
        **kwargs,
    ):
        del n_permutations, num_inputs, permutation_starting_idx  # unused
        super().__init__(**kwargs, hidden_size=hidden_size, num_inputs=3)
        self.n_sequence = n_sequence
        self.obj_embedding = nn.Embedding(n_objects, hidden_size)
        self.seq_embedding = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def embed_inputs(self, inputs: torch.Tensor):
        o = inputs.size(-1) - self.n_sequence
        rgb, sequence = torch.split(inputs, [o, self.n_sequence], dim=-1)
        rgb = (
            rgb.reshape(-1, *Sequence.miniworld_obs_shape)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        rgb = super().embed_inputs(rgb)
        sequence = self.obj_embedding(sequence.long())
        _, [sequence] = self.seq_embedding(sequence)
        return rgb + sequence


class MLPBase(Network):
    def __init__(self, num_inputs: int, recurrent: bool, hidden_size: int, **kwargs):
        super(MLPBase, self).__init__(
            recurrent=recurrent,
            recurrent_input_size=num_inputs,
            hidden_size=hidden_size,
            **kwargs,
        )

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        rnn_hxs: torch.Tensor,
        tasks: Optional[torch.Tensor] = None,
    ):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)
        if tasks is not None and self.task_embedding is not None:
            tasks = self.task_embedding(tasks)
            hidden_actor = hidden_actor + tasks
            hidden_critic = hidden_critic + tasks

        values = self.critic_linear.forward(hidden_critic).squeeze(-1)
        return values, hidden_actor, rnn_hxs
