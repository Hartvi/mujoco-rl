"""Simple Gymnasium bowling environment driven by the XML pincer."""
from __future__ import annotations

from typing import Any
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from bowling_scene import make_bowling_xml
from pincer_controller import PincerController


class BowlingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, max_steps: int = 500, num_pins: int = 10):
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.num_pins = num_pins
        self.model = mujoco.MjModel.from_xml_string(make_bowling_xml(include_pincer=True))
        self.data = mujoco.MjData(self.model)
        self._viewer: Any = None
        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self._previous_fallen = 0
        self._last_action_time = 0.0
        self._last_action_dt = 0.0
        self.task = "bowl the pins with the pincer"
        self._ee = PincerController(self.model, self.data)
        self._pin_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"pin_{i}") for i in range(1, num_pins + 1)]

        self.action_space = spaces.Box(
            low=np.array([-0.001] * 3 + [-0.01] * 3 + [-0.001], dtype=np.float32),
            high=np.array([0.001] * 3 + [0.01] * 3 + [0.001], dtype=np.float32),
        )
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
        })

    def _fallen_pins(self) -> int:
        return sum(float(self.data.xmat[body_id].reshape(3, 3)[2, 2]) < 0.75 for body_id in self._pin_ids)

    def _relevant_pin_distance(self, fallen: int) -> float:
        pincer_position = self.data.xpos[self._ee.body_id]
        pin_positions = np.asarray([self.data.xpos[body_id] for body_id in self._pin_ids])
        fallen_mask = np.asarray([self.data.xmat[body_id].reshape(3, 3)[2, 2] < 0.75 for body_id in self._pin_ids])
        candidates = pin_positions[fallen_mask] if fallen else pin_positions
        return float(np.min(np.linalg.norm(candidates - pincer_position, axis=1)))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self._ee.reset()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._previous_fallen = 0
        self._last_action_time = float(self.data.time)
        self._last_action_dt = 0.0
        return self._get_observation(), {"fallen_pins": 0, "task": self.task}

    def _get_observation(self) -> dict[str, np.ndarray]:
        return {"observation.state": np.concatenate((self._ee.observation(), [self._last_action_dt])).astype(np.float32)}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(action.astype(np.float32)):
            raise ValueError(f"Action outside space: {action}")
        action_time = float(self.data.time)
        self._ee.apply_delta(action)
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        self._ee.hold_pose()
        self._last_action_dt = float(self.data.time - action_time)
        self._last_action_time = float(self.data.time)
        self._step_count += 1
        fallen = self._fallen_pins()
        terminated = fallen == self.num_pins
        truncated = self._step_count >= self.max_steps
        pin_distance = self._relevant_pin_distance(fallen)
        distance_reward = -pin_distance
        newly_fallen = max(0, fallen - self._previous_fallen)
        fallen_reward = float(newly_fallen)
        self._previous_fallen = fallen
        open_close_reward = -10 * pin_distance * abs(float(action[6]))
        reward = distance_reward + fallen_reward + open_close_reward
        if self.render_mode == "human":
            self.render()
        return self._get_observation(), reward, terminated, truncated, {
            "fallen_pins": fallen, "newly_fallen": newly_fallen, "success": terminated, "task": self.task,
            "reward.distance": distance_reward,
            "reward.fallen_pins": fallen_reward,
            "reward.open_close": open_close_reward,
            "distance.relevant_pin": pin_distance,
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
        if self._viewer is not None:
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
            env.step(np.zeros(7, dtype=np.float32))
            time.sleep(1.0 / 60.0)
    finally:
        env.close()
