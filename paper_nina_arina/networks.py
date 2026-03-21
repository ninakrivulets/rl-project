import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


def reset_module(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, log_std_min, log_std_max):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.mu = nn.Linear(prev_dim, action_dim)
        self.log_std = nn.Linear(prev_dim, action_dim)
        reset_module(self)

    def forward(self, obs):
        hidden = self.backbone(obs)
        mu = self.mu(hidden)
        log_std = self.log_std(hidden)
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (log_std + 1.0) * (
            self.log_std_max - self.log_std_min
        )
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        raw_action = dist.rsample()
        squashed_action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(
            1 - squashed_action.pow(2) + 1e-6
        )
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        squashed_mu = torch.tanh(mu)
        return squashed_action, log_prob, squashed_mu


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims):
        super().__init__()
        self.q = build_mlp(obs_dim + action_dim, hidden_dims, 1)
        reset_module(self)

    def forward(self, obs, actions):
        return self.q(torch.cat([obs, actions], dim=-1))
