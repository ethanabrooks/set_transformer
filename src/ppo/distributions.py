import torch
import torch.nn as nn

from ppo.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical


class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions: torch.Tensor):
        log_prob: torch.Tensor = super().log_prob(actions.squeeze(-1))
        return log_prob.view(actions.size(0), -1).sum(-1)

    def mode(self):
        probs: torch.Tensor = self.probs
        return probs.argmax(dim=-1)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions: torch.Tensor):
        log_prob: torch.Tensor = super().log_prob(actions)
        return log_prob.sum(-1)

    def mode(self):
        return self.mean

    @property
    def probs(self):
        return None


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions: torch.Tensor):
        return super().log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        gt: torch.Tensor = torch.gt(self.probs, 0.5)
        return gt.float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x: torch.Tensor):
        action_mean: torch.Tensor = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd: torch.Tensor = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
