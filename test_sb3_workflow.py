from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from evaluate_pincer_sb3 import load_sb3_policy
from train_pincer_sb3 import train


class StableBaselinesWorkflowTest(unittest.TestCase):
    def test_short_training_saves_reloadable_model_and_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            args = argparse.Namespace(
                total_timesteps=16,
                n_steps=16,
                n_envs=1,
                start_method="forkserver",
                batch_size=8,
                epochs=1,
                episode_max_steps=8,
                learning_rate=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip=0.2,
                entropy_coef=0.01,
                value_coef=0.5,
                device="cpu",
                seed=7,
                output_dir=str(output),
                tensorboard_log="",
                resume="",
                vecnormalize="",
                checkpoint_freq=16,
                eval_freq=16,
                eval_episodes=1,
                log_interval=1,
                progress_bar=False,
            )

            train(args)

            model = output / "final_model.zip"
            stats = output / "vecnormalize.pkl"
            self.assertTrue(model.exists())
            self.assertTrue(stats.exists())
            self.assertTrue((output / "config.json").exists())
            policy = load_sb3_policy(str(model), str(stats), "cpu", True)
            self.assertTrue(callable(policy))


if __name__ == "__main__":
    unittest.main()
