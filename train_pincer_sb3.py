"""Train the pincer policy with Stable-Baselines3 PPO."""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from bowling_simple import BowlingEnv


def train(args: argparse.Namespace) -> None:
    if args.n_steps % args.batch_size != 0:
        raise ValueError(
            "For one environment, --batch-size must divide --n-steps "
            f"cleanly ({args.n_steps} % {args.batch_size} != 0)."
        )

    set_random_seed(args.seed, using_cuda=args.device.startswith("cuda"))
    env = Monitor(
        gym.wrappers.FlattenObservation(
            BowlingEnv(max_steps=args.episode_max_steps)
        ),
        info_keywords=("fallen_pins", "success"),
    )
    env.reset(seed=args.seed)

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    tensorboard_log = args.tensorboard_log or None
    if tensorboard_log is not None:
        Path(tensorboard_log).mkdir(parents=True, exist_ok=True)

    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    }
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip,
        vf_coef=args.value_coef,
        max_grad_norm=1.0,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=tensorboard_log,
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            log_interval=args.log_interval,
            progress_bar=args.progress_bar,
        )
        model.save(checkpoint)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--episode-max-steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="pincer_sb3")
    parser.add_argument("--tensorboard-log", default="")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
