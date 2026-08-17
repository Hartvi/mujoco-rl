"""Standalone SmolVLA inference without ROS2.

Inputs:
  observation.images.laptop: RGB uint8 image, HWC layout
  observation.images.phone:  RGB uint8 image, HWC layout
  observation.state:         six-element float vector
  task:                      task/instruction string

Output:
  action:                    six-element float vector
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Never

import numpy as np
import torch

# Prefer the checkout next to this script over an unrelated site-packages
# installation. The LeRobot package imports many optional policies at package
# import time, so keep those imports lazy until a model is actually requested.
_LOCAL_LEROBOT_SRC: Path = Path(__file__).resolve().parent / "lerobot" / "src"
if _LOCAL_LEROBOT_SRC.is_dir() and str(_LOCAL_LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_LOCAL_LEROBOT_SRC))


IMAGE_KEYS = ("observation.images.laptop", "observation.images.phone")
STATE_KEY = "observation.state"
ACTION_KEY = "action"
STATE_DIM = 6
ACTION_DIM = 6


def _choose_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    result = torch.device(device)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available")
    if result.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but MPS is not available")
    return result


def _validate_image(name: str, image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"{name} must have shape (height, width, 3), got {image.shape}"
        )
    if image.dtype != np.uint8:
        if (
            np.issubdtype(image.dtype, np.floating)
            and image.min() >= 0
            and image.max() <= 1
        ):
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _image_tensor(name: str, image: np.ndarray) -> torch.Tensor:
    """Convert an RGB HWC image to the CHW float format expected by LeRobot."""
    image = _validate_image(name, image)
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0


class SmolVLAInference:
    """Load a SmolVLA checkpoint and predict one action at a time."""

    def __init__(self, policy_path: str, device: str | None = "auto") -> None:
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy

        self.device: torch.device = _choose_device(device)
        self.policy_path: str = policy_path
        self.policy = SmolVLAPolicy.from_pretrained(policy_path)
        self.policy.to(self.device)
        self.policy.eval()
        # The direct processor factory reads the device from the policy config;
        # unlike the serialized-pipeline path it does not consume overrides.
        self.policy.config.device = str(self.device)

        processor_overrides: dict[str, dict[str, str]] = {
            "device_processor": {"device": str(self.device)},
        }
        try:
            self.preprocess, self.postprocess = make_pre_post_processors(
                policy_cfg=self.policy.config,
                pretrained_path=policy_path,
                preprocessor_overrides=processor_overrides,
            )
        except FileNotFoundError as exc:
            # Some checkpoints have model/config files but were not exported
            # with LeRobot's processor JSON files. Build the standard SmolVLA
            # pipelines directly from the policy config in that case.
            if "policy_preprocessor.json" not in str(exc):
                raise
            self.preprocess, self.postprocess = make_pre_post_processors(
                policy_cfg=self.policy.config,
                preprocessor_overrides=processor_overrides,
            )
        self._validate_model_schema()

    def _validate_model_schema(self) -> None:
        input_features = self.policy.config.input_features
        output_features = self.policy.config.output_features
        missing: list[str] = [key for key in IMAGE_KEYS if key not in input_features]
        if missing:
            raise ValueError(f"Checkpoint is missing image inputs: {missing}")
        if STATE_KEY not in input_features:
            raise ValueError(f"Checkpoint is missing {STATE_KEY!r}")
        if ACTION_KEY not in output_features:
            raise ValueError(f"Checkpoint is missing {ACTION_KEY!r}")

        state_shape: tuple[Never] = tuple(
            getattr(input_features[STATE_KEY], "shape", ())
        )
        action_shape: tuple[Never] = tuple(
            getattr(output_features[ACTION_KEY], "shape", ())
        )
        if state_shape and state_shape != (STATE_DIM,):
            raise ValueError(f"Expected state shape {(STATE_DIM,)}, got {state_shape}")
        if action_shape and action_shape != (ACTION_DIM,):
            raise ValueError(
                f"Expected action shape {(ACTION_DIM,)}, got {action_shape}"
            )

    def reset(self) -> None:
        """Clear SmolVLA temporal/action queues at episode start."""
        self.policy.reset()

    @torch.inference_mode()
    def predict(
        self,
        *,
        laptop_image: np.ndarray,
        phone_image: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state must have shape {(STATE_DIM,)}, got {state.shape}")

        observation: dict[str, Any] = {
            "task": task,
            STATE_KEY: torch.from_numpy(state.copy()),
            IMAGE_KEYS[0]: _image_tensor(IMAGE_KEYS[0], laptop_image),
            IMAGE_KEYS[1]: _image_tensor(IMAGE_KEYS[1], phone_image),
        }
        processed = self.preprocess(observation)
        # A checkpoint without serialized processor JSON uses the direct
        # factory fallback; make the final device placement explicit for every
        # tensor, including tokenized language inputs.
        processed = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in processed.items()
        }
        action = self.postprocess(self.policy.select_action(processed))
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.asarray(action, dtype=np.float32).squeeze()
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"Model returned action shape {action.shape}, expected {(ACTION_DIM,)}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"Model returned non-finite action: {action}")
        return action


def _load_rgb(path: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("The CLI requires Pillow: pip install pillow") from exc
    with Image.open(Path(path)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one standalone SmolVLA inference step"
    )
    parser.add_argument(
        "--policy", required=True, help="Checkpoint path or Hugging Face model ID"
    )
    parser.add_argument("--laptop-image", required=True)
    parser.add_argument("--phone-image", required=True)
    parser.add_argument(
        "--state", required=True, nargs=STATE_DIM, type=float, metavar="S"
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    args: argparse.Namespace = parser.parse_args()

    model = SmolVLAInference(args.policy, args.device)
    model.reset()
    action: np.ndarray[tuple[Any, ...], np.dtype[Any]] = model.predict(
        laptop_image=_load_rgb(args.laptop_image),
        phone_image=_load_rgb(args.phone_image),
        state=np.asarray(args.state, dtype=np.float32),
        task=args.task,
    )
    print("action:", action.tolist())


if __name__ == "__main__":
    main()
