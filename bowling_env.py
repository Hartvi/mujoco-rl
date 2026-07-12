"""Gymnasium bowling environment with a simulated Panda arm."""
from __future__ import annotations

from typing import Any
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from bowling_scene import make_bowling_xml
from panda_controller import PandaController


class BowlingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, max_steps: int = 500):
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.model = mujoco.MjModel.from_xml_string(make_bowling_xml())
        self.data = mujoco.MjData(self.model)
        self._viewer: Any = None
        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self.task = "bowl the pins"
        self._ee = PandaController(self.model, self.data)
        self._pin_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"pin_{i}") for i in range(1, 11)]

        self.action_space = spaces.Box(
            low=np.array([-0.05] * 3 + [-0.2] * 3, dtype=np.float32),
            high=np.array([0.05] * 3 + [0.2] * 3, dtype=np.float32),
        )
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "observation.images.laptop": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
            "observation.images.phone": spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
        })

    def _fallen_pins(self) -> int:
        fallen = 0
        for body_id in self._pin_ids:
            z_axis = self.data.xmat[body_id].reshape(3, 3)[:, 2]
            if z_axis[2] < 0.75:
                fallen += 1
        return fallen

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self._ee.reset()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_observation(), {"fallen_pins": 0, "task": self.task}

    def _get_observation(self) -> dict[str, np.ndarray]:
        return {
            "observation.state": self._ee.observation(),
            "observation.images.laptop": self._render_camera("laptop_camera"),
            "observation.images.phone": self._render_camera("phone_camera"),
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(action.astype(np.float32)):
            raise ValueError(f"Action outside space: {action}")

        self._ee.apply_delta(action)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        fallen = self._fallen_pins()
        terminated = fallen == 10
        truncated = self._step_count >= self.max_steps
        reward = float(fallen) - 0.01 * float(np.dot(action, action))

        if self.render_mode == "human":
            self.render()
        return self._get_observation(), reward, terminated, truncated, {
            "fallen_pins": fallen,
            "success": terminated,
            "task": self.task,
        }

    def _render_camera(self, camera_name: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render().copy()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_camera("laptop_camera")
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None and self.render_mode == "human":
            self._viewer.close()
        if self._renderer is not None:
            self._renderer.close()
        self._viewer = None
        self._renderer = None


if __name__ == "__main__":
    env = BowlingEnv(render_mode="human")
    try:
        env.reset()
        while env._viewer is None or env._viewer.is_running():
            env.step(np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
            time.sleep(1.0 / 60.0)
    finally:
        env.close()
