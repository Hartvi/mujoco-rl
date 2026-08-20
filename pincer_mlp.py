from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.serialization import MAP_LOCATION


class PincerMLP(nn.Module):
    """Actor-critic MLP for the pincer and fixed-order pin state."""

    action_scale: torch.Tensor
    action_bias: torch.Tensor
    observation_mean: torch.Tensor
    observation_var: torch.Tensor
    observation_count: torch.Tensor

    def __init__(
        self,
        observation_dim: int = 148,
        action_dim: int = 7,
        hidden_dim: int = 256,
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        low = (
            np.full(action_dim, -1, dtype=np.float32)
            if action_low is None
            else action_low
        )
        high = (
            np.full(action_dim, 1, dtype=np.float32)
            if action_high is None
            else action_high
        )
        self.register_buffer("action_scale", torch.as_tensor((high - low) / 2))
        self.register_buffer("action_bias", torch.as_tensor((high + low) / 2))
        self.register_buffer("observation_mean", torch.zeros(observation_dim))
        self.register_buffer("observation_var", torch.ones(observation_dim))
        self.register_buffer("observation_count", torch.tensor(1e-4))
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    @staticmethod
    def flatten_observation(
        observation: Mapping[str, np.ndarray] | np.ndarray,
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        state = (
            observation["observation.state"]
            if isinstance(observation, Mapping)
            else observation
        )
        return np.asarray(state, dtype=np.float32).reshape(-1)

    @torch.no_grad()
    def prepare_observation(
        self, observation: dict[str, np.ndarray], *, update: bool = False
    ) -> torch.Tensor:
        state: torch.Tensor = torch.as_tensor(
            self.flatten_observation(observation),
            dtype=torch.float32,
            device=self.action_scale.device,
        )
        if update:
            count: torch.Tensor = self.observation_count
            total: torch.Tensor = count + 1.0
            delta: torch.Tensor = state - self.observation_mean
            new_mean: torch.Tensor = self.observation_mean + delta / total
            old_m2: torch.Tensor = self.observation_var * count
            new_m2: torch.Tensor = old_m2 + delta * (state - new_mean)
            self.observation_mean.copy_(new_mean)
            self.observation_var.copy_(new_m2 / total)
            self.observation_count.copy_(total)
        normalized: torch.Tensor = (state - self.observation_mean) / torch.sqrt(
            self.observation_var + 1e-8
        )
        return torch.clamp(normalized, -10.0, 10.0)

    def distribution(self, state: torch.Tensor) -> Normal:
        mean = self.actor(state)
        return Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)

    def action_and_value(
        self, state: torch.Tensor, raw_action: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, Any, torch.Tensor]:
        dist: Normal = self.distribution(state)
        raw: torch.Tensor = dist.rsample() if raw_action is None else raw_action
        squashed: torch.Tensor = torch.tanh(raw)
        action: torch.Tensor = self.action_bias + self.action_scale * squashed
        log_prob: torch.Tensor = dist.log_prob(raw).sum(-1) - torch.log(
            1 - squashed.square() + 1e-6
        ).sum(-1)
        return action, log_prob, self.value(state), raw

    @torch.no_grad()
    def act(
        self, observation: dict[str, np.ndarray], deterministic: bool = False
    ) -> np.ndarray[tuple[Any, ...], np.dtype[np.floating]]:
        state: torch.Tensor = self.prepare_observation(observation).unsqueeze(0)
        dist: Normal = self.distribution(state)
        raw: torch.Tensor = dist.mean if deterministic else dist.sample()
        return (
            (self.action_bias + self.action_scale * torch.tanh(raw))[0]
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location: MAP_LOCATION = "cpu") -> None:
        self.load_state_dict(
            torch.load(path, map_location=map_location, weights_only=True),
            strict=False,
        )
