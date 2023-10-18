from collections import Counter
from dataclasses import asdict

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

import data.base
from metrics import get_metrics
from tabular.value_iteration import ValueIteration, round_tensor


class RLData(data.base.RLData):
    def __init__(
        self,
        grid_world_args: dict,
        pi_lower_bound: float,
        n_pi_bins: int,
        n_data: int,
        omit_states_actions: int,
        seed: int,
        stop_at_rmse: float,
    ):
        self.omit_states_actions = omit_states_actions
        self.stop_at_rmse = stop_at_rmse
        # 2D deltas for up, down, left, right
        grid_world = ValueIteration(**grid_world_args, n_tasks=n_data, seed=seed)
        self.grid_world = grid_world
        A = len(grid_world.deltas)
        S = grid_world.n_states
        B = n_data

        alpha = torch.ones(A)
        Pi = torch.distributions.Dirichlet(alpha).sample((B, S))  # random policies
        assert [*Pi.shape] == [B, S, A]

        # Compute the policy conditioned transition function
        Pi = round_tensor(Pi, n_pi_bins) / n_pi_bins
        Pi = Pi.float()
        Pi = torch.clamp(Pi, pi_lower_bound, 1)
        Pi = Pi / Pi.sum(-1, keepdim=True)

        print("Policy evaluation...")
        V = torch.stack(grid_world.evaluate_policy_iteratively(Pi, stop_at_rmse))
        self.V = V
        self._max_n_bellman = len(V) - 1

        states = torch.arange(S).repeat_interleave(A)
        states = states[None].tile(B, 1)
        actions = torch.arange(A).repeat(S)
        actions = actions[None].tile(B, 1)
        next_states, rewards, _, _ = grid_world.step_fn(states, actions)

        # sample n_bellman -- number of steps of policy evaluation
        self._input_bellman = input_bellman = torch.randint(
            0, self._max_n_bellman, (B, 1)
        ).tile(1, A * S)

        n_bellman = [input_bellman + o for o in range(len(V))]
        n_bellman = [torch.clamp(o, 0, self.max_n_bellman) for o in n_bellman]
        arange = torch.arange(B)[:, None]
        action_probs = Pi[arange, states]
        V = [V[o, arange, states] for o in n_bellman]

        self._values = [torch.Tensor(v) for v in V]
        self._continuous = torch.Tensor(action_probs)
        discrete = [
            states[..., None],
            actions[..., None],
            next_states[..., None],
            rewards[..., None],
        ]
        self._discrete = torch.cat(discrete, -1).long()

        perm = torch.rand(B, S * A).argsort(dim=1)

        def shuffle(x: torch.Tensor):
            p = perm
            while p.dim() < x.dim():
                p = p[..., None]

            return torch.gather(x, 1, p.expand_as(x))

        if omit_states_actions > 0:
            *self._values, self._continuous, self._discrete = [
                shuffle(x)[:, omit_states_actions:]
                for x in [*self._values, self._continuous, self._discrete]
            ]
            self.input_n_bellman = input_bellman[:, omit_states_actions:].cuda()

    @property
    def continuous(self) -> torch.Tensor:
        return self._continuous

    @property
    def discrete(self) -> torch.Tensor:
        return self._discrete

    @property
    def input_bellman(self) -> torch.Tensor:
        return self._input_bellman

    @property
    def max_n_bellman(self):
        return self._max_n_bellman

    @property
    def values(self) -> list[torch.Tensor]:
        return self._values

    def get_metrics(
        self,
        idxs: torch.Tensor,
        loss: Tensor,
        outputs: Tensor,
        targets: Tensor,
        accuracy_threshold: float,
    ):
        metrics = get_metrics(
            loss=loss,
            outputs=outputs,
            targets=targets,
            accuracy_threshold=accuracy_threshold,
        )
        metrics = asdict(metrics)
        if self.omit_states_actions == 0:
            values = outputs[:, :: len(self.grid_world.deltas)]
            Pi = self.grid_world.improve_policy(values, idxs=idxs)
            values = self.grid_world.evaluate_policy_iteratively(Pi, self.stop_at_rmse)
            metrics.update(improved_policy_value=values[-1].mean().item())
        return metrics

    def evaluate(
        self,
        bellman_delta: int,
        iterations: int,
        n_batch: int,
        net: nn.Module,
        **metrics_args,
    ):
        net.eval()
        counter = Counter()
        loader = DataLoader(self, batch_size=n_batch, shuffle=False)
        with torch.no_grad():
            for x in loader:
                (idxs, input_n_bellman, action_probs, discrete, *values) = [
                    x.cuda() for x in x
                ]
                max_n_bellman = len(values) - 1
                v1 = values[0]
                final_outputs = torch.zeros_like(v1)
                for i in range(iterations):
                    outputs: torch.Tensor
                    loss: torch.Tensor
                    outputs, loss = net.forward(
                        v1=v1,
                        action_probs=action_probs,
                        discrete=discrete,
                        targets=values[min((i + 1) * bellman_delta, max_n_bellman)],
                    )
                    v1 = outputs.squeeze(-1)
                    mask = (input_n_bellman + i * bellman_delta) < max_n_bellman
                    final_outputs[mask] = v1[mask]

                metrics = self.get_metrics(
                    idxs=idxs,
                    loss=loss,
                    outputs=final_outputs,
                    targets=values[iterations],
                    **metrics_args,
                )
                counter.update(metrics)
        return {k: v / len(loader) for k, v in counter.items()}
