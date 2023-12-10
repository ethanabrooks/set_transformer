from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from gym import Space
from torch.optim import Optimizer

from ppo.distributions import Bernoulli, Categorical, DiagGaussian
from ppo.networks import CNNBase, CNNWithTaskEmbedding
from ppo.rollout_storage import RolloutStorage, Sample


@dataclass
class ActMetadata:
    log_probs: torch.Tensor
    probs: torch.Tensor
    rnn_hxs: torch.Tensor
    value: torch.Tensor


class Agent(nn.Module):
    def __init__(self, action_space: Space, obs_shape: tuple[int, ...], **kwargs):
        super(Agent, self).__init__()
        self.base: CNNBase = CNNBase(num_inputs=obs_shape[0], **kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(
        self,
        inputs: torch.Tensor,
        rnn_hxs: torch.Tensor,
        masks: torch.Tensor,
    ):
        raise NotImplementedError

    def act(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        rnn_hxs: torch.Tensor,
        deterministic: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, ActMetadata]:
        value, actor_features, rnn_hxs = self.base.forward(
            inputs=inputs, rnn_hxs=rnn_hxs, masks=masks, **kwargs
        )
        dist = self.dist.forward(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        return action, ActMetadata(
            log_probs=action_log_probs, probs=dist.probs, rnn_hxs=rnn_hxs, value=value
        )

    def get_value(
        self, inputs: torch.Tensor, masks: torch.Tensor, rnn_hxs: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        value, _, _ = self.base.forward(
            inputs=inputs, rnn_hxs=rnn_hxs, masks=masks, **kwargs
        )
        return value

    def evaluate_actions(
        self,
        action: torch.Tensor,
        obs: torch.Tensor,
        masks: torch.Tensor,
        rnn_hxs: torch.Tensor,
        **kwargs,
    ):
        value, actor_features, rnn_hxs = self.base.forward(
            inputs=obs, rnn_hxs=rnn_hxs, masks=masks, **kwargs
        )
        dist = self.dist.forward(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return dist_entropy, ActMetadata(
            log_probs=action_log_probs, probs=dist.probs, rnn_hxs=rnn_hxs, value=value
        )

    def update(
        self,
        clip_param: float,
        entropy_coef: float,
        num_mini_batch: int,
        optimizer: Optimizer,
        ppo_epoch: int,
        rollouts: RolloutStorage,
        value_loss_coef: float,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
    ):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for _ in range(ppo_epoch):
            if self.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, num_mini_batch
                )
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, num_mini_batch
                )

            sample: Sample
            for sample in data_generator:
                # Reshape to do in a single forward pass for all steps
                condition_on_task = isinstance(self.base, CNNWithTaskEmbedding)
                dist_entropy, action_metadata = self.evaluate_actions(
                    action=sample.actions,
                    masks=sample.masks,
                    obs=sample.obs,
                    rnn_hxs=sample.rnn_hxs,
                    **(dict(tasks=sample.tasks) if condition_on_task else {}),
                )

                ratio = torch.exp(
                    action_metadata.log_probs - sample.old_action_log_probs
                )
                adv_targ = sample.adv_targets
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if use_clipped_value_loss:
                    values = action_metadata.value
                    value_preds = sample.value_preds
                    value_diff: torch.Tensor = values - value_preds
                    value_pred_clipped = value_preds + value_diff.clamp(
                        -clip_param, clip_param
                    )
                    returns = sample.returns
                    value_losses: torch.Tensor = values - returns
                    value_losses = value_losses.pow(2)
                    value_losses_clipped = (value_pred_clipped - returns).pow(2)
                    value_loss = (
                        0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss: torch.Tensor = returns - values
                    value_loss = 0.5 * value_loss.pow(2).mean()

                optimizer.zero_grad()
                (
                    value_loss * value_loss_coef
                    + action_loss
                    - dist_entropy * entropy_coef
                ).backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = ppo_epoch * num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
