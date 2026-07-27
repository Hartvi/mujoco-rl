"""Primary Stable-Baselines3 PPO training entry point for pincer bowling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from bowling_simple import BowlingEnv
from pincer_controller import PincerController


REWARD_KEYS = (
    "reward.distance",
    "reward.rotation",
    "reward.ground_clearance",
    "reward.fallen_pins",
    "reward.open_close",
    "reward.action",
    "reward.pin_touch",
    "reward.success",
)


def make_env(episode_max_steps: int, seed: int, monitor_path: Path | None):
    def factory():
        env = gym.wrappers.FlattenObservation(
            BowlingEnv(max_steps=episode_max_steps)
        )
        env = Monitor(
            env,
            filename=str(monitor_path) if monitor_path is not None else None,
            info_keywords=("fallen_pins", "success"),
        )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return factory


class RewardLoggingCallback(BaseCallback):
    """Publish mean reward components and task metrics to SB3's logger."""

    def __init__(self) -> None:
        super().__init__()
        self.values = {key: [] for key in REWARD_KEYS}
        self.fallen: list[float] = []
        self.distances: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in REWARD_KEYS:
                if key in info:
                    self.values[key].append(float(info[key]))
            if "fallen_pins" in info:
                self.fallen.append(float(info["fallen_pins"]))
            if "distance.relevant_pin" in info:
                self.distances.append(float(info["distance.relevant_pin"]))
        return True

    def _on_rollout_end(self) -> None:
        for key, values in self.values.items():
            if values:
                self.logger.record(key, float(np.mean(values)))
                values.clear()
        if self.fallen:
            self.logger.record("task/mean_fallen_pins", float(np.mean(self.fallen)))
            self.logger.record("task/max_fallen_pins", max(self.fallen))
            self.fallen.clear()
        if self.distances:
            self.logger.record(
                "task/mean_target_distance", float(np.mean(self.distances))
            )
            self.distances.clear()


class SaveBestVecNormalizeCallback(BaseCallback):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    def _on_step(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.parent.eval_env.save(str(self.path))
        return True


def _resolve_resume_stats(resume: Path, explicit: str) -> Path | None:
    if explicit:
        return Path(explicit)
    candidates = (
        resume.parent / "vecnormalize.pkl",
        resume.parent / f"{resume.stem}_vecnormalize.pkl",
    )
    return next((path for path in candidates if path.exists()), None)


def train(args: argparse.Namespace) -> None:
    rollout_size = args.n_steps * args.n_envs
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            "--batch-size must divide --n-steps * --n-envs cleanly "
            f"({rollout_size} % {args.batch_size} != 0)."
        )
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")

    set_random_seed(args.seed, using_cuda=args.device.startswith("cuda"))
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    best_dir = output_dir / "best"
    for path in (output_dir, checkpoint_dir, best_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_factories = [
        make_env(
            args.episode_max_steps,
            args.seed + rank,
            output_dir / f"train_monitor_{rank}",
        )
        for rank in range(args.n_envs)
    ]
    if args.n_envs == 1:
        train_base = DummyVecEnv(train_factories)
    else:
        train_base = SubprocVecEnv(
            train_factories,
            start_method=args.start_method,
        )
    resume = Path(args.resume) if args.resume else None
    stats_path = _resolve_resume_stats(resume, args.vecnormalize) if resume else None
    if stats_path is not None:
        train_env = VecNormalize.load(str(stats_path), train_base)
        train_env.training = True
        train_env.norm_reward = False
    else:
        train_env = VecNormalize(train_base, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_base = DummyVecEnv([
        make_env(args.episode_max_steps, args.seed + 10_000, output_dir / "eval_monitor")
    ])
    eval_env = VecNormalize(eval_base, training=False, norm_obs=True, norm_reward=False)

    policy_kwargs = {
        "activation_fn": torch.nn.Tanh,
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    }
    tensorboard_log = (
        str(output_dir / "tensorboard")
        if args.tensorboard_log == "auto"
        else args.tensorboard_log or None
    )
    if tensorboard_log is not None:
        Path(tensorboard_log).mkdir(parents=True, exist_ok=True)
    if resume is not None:
        model = PPO.load(str(resume), env=train_env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip,
            ent_coef=args.entropy_coef,
            vf_coef=args.value_coef,
            max_grad_norm=1.0,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=tensorboard_log,
        )

    callbacks = CallbackList([
        RewardLoggingCallback(),
        CheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="pincer",
            save_vecnormalize=True,
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            callback_on_new_best=SaveBestVecNormalizeCallback(
                best_dir / "vecnormalize.pkl"
            ),
            best_model_save_path=str(best_dir),
            log_path=str(output_dir / "evaluations"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            verbose=1,
        ),
    ])

    config = vars(args).copy()
    config["action_semantics"] = [
        "vx", "vy", "vz", "wx", "wy", "wz", "jaw_velocity"
    ]
    config["physical_rate_limits"] = {
        "translation_m_per_s": PincerController.MAX_LINEAR_SPEED,
        "rotation_rad_per_s": PincerController.MAX_ANGULAR_SPEED,
        "jaw_m_per_s": PincerController.MAX_DISTANCE_SPEED,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=args.log_interval,
            progress_bar=args.progress_bar,
            reset_num_timesteps=resume is None,
        )
        model.save(str(output_dir / "final_model"))
        train_env.save(str(output_dir / "vecnormalize.pkl"))
    finally:
        train_env.close()
        eval_env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="parallel rollout workers; use 1 for DummyVecEnv debugging",
    )
    parser.add_argument(
        "--start-method",
        choices=("forkserver", "spawn", "fork"),
        default="forkserver",
        help="multiprocessing start method used by SubprocVecEnv",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--episode-max-steps", type=int, default=1500)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/pincer_sb3")
    parser.add_argument(
        "--tensorboard-log",
        default="auto",
        help="TensorBoard directory; 'auto' uses OUTPUT_DIR/tensorboard, empty disables it",
    )
    parser.add_argument("--resume", default="", help="SB3 model zip to resume")
    parser.add_argument(
        "--vecnormalize",
        default="",
        help="normalization statistics for --resume; inferred when omitted",
    )
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--progress-bar", action="store_true")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
