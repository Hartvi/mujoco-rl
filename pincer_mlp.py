from __future__ import annotations
from collections.abc import Mapping
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class PincerMLP(nn.Module):
    """Actor-critic MLP for the pincer and fixed-order pin state."""

    def __init__(self, observation_dim=148, action_dim=7, hidden_dim=256, action_low=None, action_high=None):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, action_dim))
        self.critic = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        low = np.full(action_dim, -1, dtype=np.float32) if action_low is None else action_low
        high = np.full(action_dim, 1, dtype=np.float32) if action_high is None else action_high
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2))
        self.register_buffer("observation_mean", torch.zeros(observation_dim))
        self.register_buffer("observation_var", torch.ones(observation_dim))
        self.register_buffer("observation_count", torch.tensor(1e-4))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    @staticmethod
    def flatten_observation(observation):
        if isinstance(observation, Mapping):
            observation = observation["observation.state"]
        return np.asarray(observation, dtype=np.float32).reshape(-1)

    @torch.no_grad()
    def prepare_observation(self, observation, *, update=False):
        state = torch.as_tensor(
            self.flatten_observation(observation),
            dtype=torch.float32,
            device=self.action_scale.device,
        )
        if update:
            count = self.observation_count
            total = count + 1.0
            delta = state - self.observation_mean
            new_mean = self.observation_mean + delta / total
            old_m2 = self.observation_var * count
            new_m2 = old_m2 + delta * (state - new_mean)
            self.observation_mean.copy_(new_mean)
            self.observation_var.copy_(new_m2 / total)
            self.observation_count.copy_(total)
        normalized = (state - self.observation_mean) / torch.sqrt(
            self.observation_var + 1e-8
        )
        return torch.clamp(normalized, -10.0, 10.0)

    def distribution(self, state):
        mean = self.actor(state)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, state):
        return self.critic(state).squeeze(-1)

    def action_and_value(self, state, raw_action=None):
        dist = self.distribution(state)
        raw = dist.rsample() if raw_action is None else raw_action
        squashed = torch.tanh(raw)
        action = self.action_bias + self.action_scale * squashed
        log_prob = dist.log_prob(raw).sum(-1) - torch.log(1 - squashed.square() + 1e-6).sum(-1)
        return action, log_prob, self.value(state), raw

    @torch.no_grad()
    def act(self, observation, deterministic=False):
        state = self.prepare_observation(observation).unsqueeze(0)
        dist = self.distribution(state)
        raw = dist.mean if deterministic else dist.sample()
        return (self.action_bias + self.action_scale * torch.tanh(raw))[0].cpu().numpy().astype(np.float32)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location="cpu"):
        self.load_state_dict(
            torch.load(path, map_location=map_location, weights_only=True),
            strict=False,
        )
