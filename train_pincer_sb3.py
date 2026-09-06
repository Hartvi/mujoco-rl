"""Primary Stable-Baselines3 PPO training entry point for pincer bowling."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast, Sequence

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
)

from bowling_env_config import (
    BowlingEnvConfig,
    ENV_TYPES,
    make_env_factory,
    parse_bool,
    resolve_env_config,
)
from bowling_scene import PinComponent
from pincer_controller import PincerController
from torch_device import resolve_device_name


class StrictParser(argparse.ArgumentParser):
    """Strict bowling parser that encodes correctness of arguments using types."""

    def parse_args(self, args: Sequence[str] | None = None) -> TrainingConfig:  # type: ignore
        namespace = super().parse_args(args)
        try:
            env_config = resolve_env_config(
                env_type=namespace.env_type,
                max_steps=namespace.episode_max_steps,
                num_pins=namespace.num_pins,
                pin_component=namespace.pin_component,
                pins_fallen=namespace.pins_fallen,
                render=namespace.render,
            )
        except ValueError as error:
            self.error(str(error))
        values = vars(namespace)
        for key in ("env_type", "num_pins", "pin_component", "pins_fallen"):
            values.pop(key)
        return TrainingConfig(env_config=env_config, **values)


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 1_000_000
    n_steps: int = 512
    n_envs: int = 4
    start_method: str = "forkserver"
    batch_size: int = 128
    epochs: int = 4
    episode_max_steps: int = 1500
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    device: str = "auto"
    seed: int = 0
    output_dir: str = "runs/pincer_sb3"
    tensorboard_log: str = "auto"
    resume: str = ""
    vecnormalize: str = ""
    checkpoint_freq: int = 50_000
    eval_freq: int = 25_000
    eval_episodes: int = 5
    log_interval: int = 1
    progress_bar: bool = False
    env_config: BowlingEnvConfig | None = None
    # Compatibility for programmatic callers of the former StrictArgs namespace.
    env_type: str = "BowlingSimple"
    num_pins: int | None = None
    pin_component: PinComponent | None = None
    pins_fallen: bool | None = None
    render: str | None = None

    def resolved_env_config(self) -> BowlingEnvConfig:
        if self.env_config is not None:
            return self.env_config
        return resolve_env_config(
            env_type=self.env_type,
            max_steps=self.episode_max_steps,
            num_pins=self.num_pins,
            pin_component=self.pin_component,
            pins_fallen=self.pins_fallen,
            render=self.render,
        )


# Kept as an import-compatible name while callers migrate to TrainingConfig.
StrictArgs = TrainingConfig


class RewardLoggingCallback(BaseCallback):
    """Publish mean reward components and task metrics to SB3's logger."""

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, list[float]] = {}
        self.fallen: list[float] = []
        self.distances: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key, value in info.items():
                if key.startswith(("reward.", "distance.")) and isinstance(
                    value, (int, float, np.number)
                ):
                    self.values.setdefault(key, []).append(float(value))
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
            self.logger.record("task/min_fallen_pins", min(self.fallen))
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
        eval_env = cast(EvalCallback, self.parent).eval_env
        cast(VecNormalize, eval_env).save(str(self.path))
        return True


def _resolve_resume_stats(resume: Path, explicit: str) -> Path | None:
    if explicit:
        return Path(explicit)
    candidates = (
        resume.parent / "vecnormalize.pkl",
        resume.parent / f"{resume.stem}_vecnormalize.pkl",
    )
    return next((path for path in candidates if path.exists()), None)


def train(args: TrainingConfig) -> None:
    env_config = args.resolved_env_config()
    rollout_size = args.n_steps * args.n_envs
    if rollout_size % args.batch_size != 0:
        raise ValueError(
            "--batch-size must divide --n-steps * --n-envs cleanly "
            f"({rollout_size} % {args.batch_size} != 0)."
        )
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")

    device = resolve_device_name(args.device)
    set_random_seed(args.seed, using_cuda=device.startswith("cuda"))
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    best_dir = output_dir / "best"
    for path in (output_dir, checkpoint_dir, best_dir):
        path.mkdir(parents=True, exist_ok=True)

    train_factories = [
        make_env_factory(
            env_config,
            seed=args.seed + rank,
            monitor_path=str(output_dir / f"train_monitor_{rank}"),
        )
        for rank in range(args.n_envs)
    ]
    train_base: VecEnv
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
        train_env = VecNormalize(
            train_base, norm_obs=True, norm_reward=False, clip_obs=10.0
        )

    eval_base = DummyVecEnv(
        [
            make_env_factory(
                env_config,
                seed=args.seed + args.n_envs,
                monitor_path=str(output_dir / "eval_monitor"),
            )
        ]
    )
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
        model = PPO.load(str(resume), env=train_env, device=device)
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
            device=device,
            tensorboard_log=tensorboard_log,
        )

    callbacks = CallbackList(
        [
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
        ]
    )

    config = asdict(args)
    config["env_config"] = env_config.to_json()
    config["device"] = device
    config["action_semantics"] = [
        "vx",
        "vy",
        "vz",
        "wx",
        "wy",
        "wz",
        "jaw_velocity",
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
    parser = StrictParser()
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
    parser.add_argument(
        "--device",
        default="auto",
        help="auto (cuda, then mps, then cpu), or an explicit torch device",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="runs/pincer_sb3")
    parser.add_argument(
        "--tensorboard-log",
        default="auto",
        help="TensorBoard directory; 'auto' uses OUTPUT_DIR/tensorboard, "
        "empty disables it",
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
    parser.add_argument(
        "--pin-component",
        default="auto",
        choices=("auto", "none", *(item.value for item in PinComponent)),
    )
    parser.add_argument(
        "--env-type",
        type=str,
        choices=tuple(ENV_TYPES),
        default="BowlingSimple",
    )
    parser.add_argument(
        "--num-pins",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--pins-fallen",
        type=parse_bool,
        default=None,
    )
    parser.add_argument("--render", type=str)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
