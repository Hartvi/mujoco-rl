"""Typed configuration and factories shared by bowling training and evaluation."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Mapping, TypeAlias

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor

from bowling_pick_up import BowlingPickUp
from bowling_scene import PinComponent
from bowling_simple import BowlingSimple


EnvType: TypeAlias = Literal["BowlingSimple", "BowlingPickUp"]


@dataclass(frozen=True)
class BowlingEnvSpec:
    env_class: type[BowlingSimple] | type[BowlingPickUp]
    supported_options: frozenset[str]


ENV_REGISTRY: dict[str, BowlingEnvSpec] = {
    "BowlingSimple": BowlingEnvSpec(
        BowlingSimple,
        frozenset({"render_mode", "max_steps", "num_pins", "pin_component"}),
    ),
    "BowlingPickUp": BowlingEnvSpec(
        BowlingPickUp,
        frozenset(
            {"render_mode", "max_steps", "num_pins", "pin_component", "pins_fallen"}
        ),
    ),
}
ENV_TYPES = {name: spec.env_class for name, spec in ENV_REGISTRY.items()}


@dataclass(frozen=True)
class BowlingEnvConfig:
    env_type: EnvType = "BowlingSimple"
    max_steps: int = 1500
    num_pins: int = 10
    pin_component: PinComponent | None = None
    pins_fallen: bool | None = None
    render: str | None = None

    def constructor_kwargs(self, render_mode: str | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "render_mode": render_mode,
            "max_steps": self.max_steps,
            "num_pins": self.num_pins,
            "pin_component": self.pin_component,
        }
        if self.env_type == "BowlingPickUp":
            kwargs["pins_fallen"] = self.pins_fallen
        return kwargs

    def to_json(self) -> dict[str, Any]:
        result = asdict(self)
        result["pin_component"] = (
            self.pin_component.value if self.pin_component is not None else None
        )
        return result


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "on"}:
        return True
    if normalized in {"0", "false", "off"}:
        return False
    raise ValueError("expected one of: 0, 1, true, false, on, off")


def parse_pin_component(value: str | PinComponent | None) -> PinComponent | None:
    if value is None or value == "none":
        return None
    if isinstance(value, PinComponent):
        return value
    return PinComponent(value)


def resolve_env_config(
    *,
    env_type: str = "BowlingSimple",
    max_steps: int = 1500,
    num_pins: int | None = None,
    pin_component: str | PinComponent | None = "auto",
    pins_fallen: str | bool | None = None,
    render: str | None = None,
) -> BowlingEnvConfig:
    if env_type not in ENV_TYPES:
        raise ValueError(f"unknown environment {env_type!r}")
    if max_steps < 1:
        raise ValueError("maximum steps must be positive")
    if env_type == "BowlingSimple":
        if pins_fallen is not None:
            raise ValueError("--pins-fallen is only valid for BowlingPickUp")
        resolved_component = (
            None if pin_component == "auto" else parse_pin_component(pin_component)
        )
        return BowlingEnvConfig(
            env_type="BowlingSimple",
            max_steps=max_steps,
            num_pins=10 if num_pins is None else num_pins,
            pin_component=resolved_component,
            pins_fallen=None,
            render=render,
        )
    resolved_component = (
        PinComponent.HEAD
        if pin_component == "auto"
        else parse_pin_component(pin_component)
    )
    return BowlingEnvConfig(
        env_type="BowlingPickUp",
        max_steps=max_steps,
        num_pins=1 if num_pins is None else num_pins,
        pin_component=resolved_component,
        pins_fallen=True if pins_fallen is None else parse_bool(pins_fallen),
        render=render,
    )


def env_config_from_mapping(
    values: Mapping[str, Any], *, max_steps: int | None = None
) -> BowlingEnvConfig:
    nested = values.get("env_config", values)
    assert isinstance(nested, Mapping)
    env_type = str(nested.get("env_type", "BowlingSimple"))
    return resolve_env_config(
        env_type=env_type,
        max_steps=int(
            max_steps
            if max_steps is not None
            else nested.get("max_steps", nested.get("episode_max_steps", 1500))
        ),
        num_pins=(int(nested["num_pins"]) if "num_pins" in nested else None),
        pin_component=nested.get("pin_component", "auto"),
        pins_fallen=(
            nested.get("pins_fallen") if env_type == "BowlingPickUp" else None
        ),
    )


def make_raw_env(
    config: BowlingEnvConfig, *, render_mode: str | None = None
) -> BowlingSimple | BowlingPickUp:
    spec: BowlingEnvSpec = ENV_REGISTRY[config.env_type]
    kwargs = config.constructor_kwargs(render_mode)
    unsupported = kwargs.keys() - spec.supported_options
    if unsupported:
        raise ValueError(f"unsupported options for {config.env_type}: {unsupported}")
    return spec.env_class(**kwargs)


def make_env_factory(
    config: BowlingEnvConfig,
    *,
    seed: int,
    monitor_path: str | None = None,
) -> Callable[[], gym.Env[Any, Any]]:
    def factory() -> gym.Env[Any, Any]:
        # Scene construction itself samples pin poses with Python's RNG.
        random.seed(seed)
        env: gym.Env[Any, Any] = gym.wrappers.FlattenObservation(
            make_raw_env(config=config, render_mode=config.render)
        )
        env = Monitor(
            env,
            filename=monitor_path,
            info_keywords=("fallen_pins", "success"),
        )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return factory
