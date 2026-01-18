from torch import nn
import torch
from typing import Type


def create_sequential(layer_sizes: list[int], activation_funct: Type[nn.Module] = nn.ReLU) -> nn.Sequential:
    modules = []

    for i in range(len(layer_sizes) - 2):
        modules.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        modules.append(activation_funct())
    modules.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

    return nn.Sequential(*modules)


class ArmNet(nn.Module):
    def __init__(self, observation_size: int, action_size: int) -> None:
        super().__init__()

        actor_layers = [observation_size, 64, 64, action_size]
        actor_activation = nn.ReLU
        self.actor = create_sequential(actor_layers, actor_activation)

        critic_layers = [observation_size, 64, 1]
        critic_activation = nn.ReLU
        self.critic = create_sequential(critic_layers, critic_activation)

        self.log_std = nn.Parameter(torch.zeros((1, action_size)), requires_grad=True)

    def forward(self, x: torch.Tensor, actions: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.actor(x)
        distribution = torch.distributions.normal.Normal(mean, torch.exp(self.log_std))

        if actions is None:
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
            return actions, log_probs.sum(dim=1), self.critic(x)

        log_probs = distribution.log_prob(actions).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return log_probs, entropy, self.critic(x)


