"""Evaluate an SB3 pincer policy and simple bowling baselines."""

from __future__ import annotations

import argparse
import csv
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bowling_simple import BowlingSimple
from torch_device import resolve_device_name


@dataclass(frozen=True)
class EpisodeResult:
    policy: str
    episode: int
    episode_return: float
    steps: int
    fallen_pins: int
    final_distance: float
    success: bool


ActionFunction = Callable[[dict[str, np.ndarray], BowlingSimple], np.ndarray]


def load_sb3_policy(
    model_path: str,
    stats_path: str,
    device: str,
    deterministic: bool,
) -> ActionFunction:
    model = PPO.load(model_path, device=resolve_device_name(device))
    probe = DummyVecEnv([lambda: gym.wrappers.FlattenObservation(BowlingSimple())])
    normalizer = VecNormalize.load(stats_path, probe)
    normalizer.training = False
    normalizer.norm_reward = False

    def act(observation: dict[str, np.ndarray], _env: BowlingSimple) -> np.ndarray:
        flat = np.asarray(observation["observation.state"], dtype=np.float32)
        normalized = cast(np.ndarray, normalizer.normalize_obs(flat[None, :]))[0]
        action, _ = model.predict(normalized, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    # Keep the normalization wrapper alive for the lifetime of the closure.
    act.normalizer = normalizer  # type: ignore[attr-defined]
    return act


def random_action(
    _observation: dict[str, np.ndarray], env: BowlingSimple
) -> np.ndarray:
    return env.action_space.sample()


def no_op_action(_observation: dict[str, np.ndarray], env: BowlingSimple) -> np.ndarray:
    return np.zeros(env.action_space.shape, dtype=np.float32)


def scripted_nearest_pin_action(
    _observation: dict[str, np.ndarray], env: BowlingSimple
) -> np.ndarray:
    pincer = env.data.xpos[env._ee.body_id]
    upright = [
        body_id
        for body_id in env._pin_ids
        if env.data.xmat[body_id].reshape(3, 3)[2, 2] >= 0.75
    ]
    action = np.zeros(7, dtype=np.float32)
    if not upright:
        return action
    offsets = np.asarray([env.data.xpos[body_id] - pincer for body_id in upright])
    offset = offsets[np.argmin(np.linalg.norm(offsets, axis=1))]
    norm = float(np.linalg.norm(offset))
    if norm > 1e-8:
        action[:3] = np.clip(offset / norm, -1.0, 1.0)
    return action


def evaluate(
    name: str,
    policy: ActionFunction,
    args: argparse.Namespace,
) -> list[EpisodeResult]:
    env = BowlingSimple(
        render_mode="human" if args.render else None,
        max_steps=args.episode_max_steps,
    )
    results = []
    try:
        for episode in range(args.episodes):
            seed = args.seed + episode
            observation, _ = env.reset(seed=seed)
            env.action_space.seed(seed)
            episode_return = 0.0
            terminated = truncated = False
            steps = 0
            info = {
                "fallen_pins": 0,
                "distance.relevant_pin": float("nan"),
                "success": False,
            }
            while not (terminated or truncated):
                action = policy(observation, env)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                steps += 1
                if args.render and args.real_time:
                    time.sleep(env.control_dt)
            result = EpisodeResult(
                policy=name,
                episode=episode + 1,
                episode_return=episode_return,
                steps=steps,
                fallen_pins=int(info["fallen_pins"]),
                final_distance=float(info["distance.relevant_pin"]),
                success=bool(info["success"]),
            )
            results.append(result)
            if not args.summary_only:
                print(
                    f"[{name}] episode={result.episode} "
                    f"return={result.episode_return:.3f} steps={result.steps} "
                    f"fallen={result.fallen_pins} "
                    f"final_distance={result.final_distance:.3f}m "
                    f"success={result.success}",
                    flush=True,
                )
    finally:
        env.close()
    return results


def print_summary(name: str, results: list[EpisodeResult]) -> None:
    returns = np.asarray([item.episode_return for item in results])
    fallen = np.asarray([item.fallen_pins for item in results])
    success = np.asarray([item.success for item in results])
    print(
        f"[{name}] episodes={len(results)} mean_return={returns.mean():.3f} "
        f"std_return={returns.std():.3f} mean_fallen={fallen.mean():.2f} "
        f"success_rate={success.mean():.1%}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/pincer_sb3/final_model.zip")
    parser.add_argument("--vecnormalize", default="runs/pincer_sb3/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-max-steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (cuda, then mps, then cpu), or an explicit torch device",
    )
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--csv", default="")
    args = parser.parse_args()
    if args.episodes < 1 or args.episode_max_steps < 1:
        parser.error("episode counts and maximum steps must be positive")

    policies = [
        (
            "checkpoint",
            load_sb3_policy(
                args.model,
                args.vecnormalize,
                args.device,
                deterministic=not args.stochastic,
            ),
        )
    ]
    if args.include_baselines:
        policies.extend(
            [
                ("scripted-nearest", scripted_nearest_pin_action),
                ("random", random_action),
                ("no-op", no_op_action),
            ]
        )

    all_results = []
    for name, policy in policies:
        results = evaluate(name, policy, args)
        all_results.extend(results)
        print_summary(name, results)

    if args.csv:
        output = Path(args.csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=asdict(all_results[0]).keys())
            writer.writeheader()
            writer.writerows(asdict(item) for item in all_results)


if __name__ == "__main__":
    main()
