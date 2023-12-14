from typing import Optional

import numpy as np
import torch
import torch.nn as nn

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

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        rnn_hxs: torch.Tensor,
        tasks: torch.Tensor = None,
    ):
        x = self.main(inputs / 255.0)
        if tasks is not None and self.task_embedding is not None:
            x = x + self.task_embedding(tasks)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)
        values = self.critic_linear.forward(x).squeeze(-1)
        return values, x, rnn_hxs


class MLPBase(Network):
    def __init__(self, num_inputs: int, recurrent: bool, hidden_size: int, **kwargs):
        super(MLPBase, self).__init__(
            recurrent=recurrent,
            recurrent_input_size=num_inputs,
            hidden_size=hidden_size,
            **kwargs
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
