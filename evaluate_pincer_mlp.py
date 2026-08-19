"""Evaluate a trained pincer MLP against the bowling environment."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from bowling_simple import BowlingSimple
from pincer_mlp import PincerMLP
from torch_device import resolve_device


@dataclass(frozen=True)
class EpisodeResult:
    episode_return: float
    steps: int
    fallen_pins: int
    final_distance: float
    success: bool


ActionFunction = Callable[[dict[str, np.ndarray], BowlingSimple], np.ndarray]


def evaluate(
    name: str,
    action_function: ActionFunction,
    *,
    episodes: int,
    episode_max_steps: int,
    seed: int,
    render: bool,
    real_time: bool,
    summary_only: bool,
) -> list[EpisodeResult]:
    env = BowlingSimple(
        render_mode="human" if render else None,
        max_steps=episode_max_steps,
    )
    results: list[EpisodeResult] = []

    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            observation, _ = env.reset(seed=episode_seed)
            env.action_space.seed(episode_seed)
            episode_return = 0.0
            terminated = truncated = False
            info = {
                "fallen_pins": 0,
                "distance.relevant_pin": float("nan"),
                "success": False,
            }
            steps = 0

            while not (terminated or truncated):
                action = action_function(observation, env)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1
                if real_time and render:
                    time.sleep(env.control_dt)

            result = EpisodeResult(
                episode_return=episode_return,
                steps=steps,
                fallen_pins=int(info["fallen_pins"]),
                final_distance=float(info["distance.relevant_pin"]),
                success=bool(info["success"]),
            )
            results.append(result)
            if not summary_only:
                print(
                    f"[{name}] episode={episode + 1} "
                    f"return={result.episode_return:.3f} "
                    f"steps={result.steps} "
                    f"fallen={result.fallen_pins} "
                    f"final_distance={result.final_distance:.3f}m "
                    f"success={result.success}",
                    flush=True,
                )
    finally:
        env.close()

    return results


def print_summary(name: str, results: list[EpisodeResult]) -> None:
    returns = np.asarray([result.episode_return for result in results])
    distances = np.asarray([result.final_distance for result in results])
    fallen = np.asarray([result.fallen_pins for result in results])
    steps = np.asarray([result.steps for result in results])
    success_rate = np.mean([result.success for result in results])

    print(
        f"[{name}] episodes={len(results)} "
        f"mean_return={returns.mean():.3f} "
        f"std_return={returns.std():.3f} "
        f"mean_final_distance={distances.mean():.3f}m "
        f"mean_fallen={fallen.mean():.2f} "
        f"success_rate={success_rate:.1%} "
        f"mean_steps={steps.mean():.1f}",
        flush=True,
    )


def load_checkpoint_policy(
    checkpoint: str,
    device: torch.device,
    *,
    deterministic: bool,
) -> ActionFunction:
    probe_env = BowlingSimple()
    try:
        model = PincerMLP(
            observation_dim=probe_env.observation_space["observation.state"].shape[0],
            action_low=probe_env.action_space.low,
            action_high=probe_env.action_space.high,
        ).to(device)
    finally:
        probe_env.close()

    model.load(checkpoint, map_location=device)
    model.eval()

    def act(observation: dict[str, np.ndarray], _env: BowlingSimple) -> np.ndarray:
        return model.act(observation, deterministic=deterministic)

    return act


def random_action(
    _observation: dict[str, np.ndarray],
    env: BowlingSimple,
) -> np.ndarray:
    return env.action_space.sample()


def no_op_action(
    _observation: dict[str, np.ndarray],
    env: BowlingSimple,
) -> np.ndarray:
    assert env.action_space.shape is not None
    return np.zeros(shape=env.action_space.shape, dtype=env.action_space.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="pincer_mlp.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (cuda, then mps, then cpu), or an explicit torch device",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample checkpoint actions instead of using the policy mean",
    )
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="also evaluate random and no-op policies",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="pace rendered evaluation according to simulated time",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="suppress individual episode lines",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.episode_max_steps < 1:
        parser.error("--episode-max-steps must be at least 1")

    device = resolve_device(args.device)
    policies = [
        (
            "checkpoint",
            load_checkpoint_policy(
                args.checkpoint,
                device,
                deterministic=not args.stochastic,
            ),
        )
    ]
    if args.include_baselines:
        policies.extend(
            [
                ("random", random_action),
                ("no-op", no_op_action),
            ]
        )

    for name, action_function in policies:
        results = evaluate(
            name,
            action_function,
            episodes=args.episodes,
            episode_max_steps=args.episode_max_steps,
            seed=args.seed,
            render=args.render,
            real_time=args.real_time,
            summary_only=args.summary_only,
        )
        print_summary(name, results)


if __name__ == "__main__":
    main()
