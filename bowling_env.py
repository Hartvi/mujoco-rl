"""MuJoCo bowling environment with ten free-jointed bowling pins."""
from __future__ import annotations
from typing import Any
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer


def _pin_xml(index: int, x: float, y: float) -> str:
    return f'''
    <body name="pin_{index}" pos="{x:.4f} {y:.4f} 0.08">
      <freejoint name="pin_{index}_free"/>
      <geom name="pin_{index}_base" type="cylinder" size="0.115 0.08" pos="0 0 0.08" rgba="0.92 0.92 0.94 1"/>
      <geom name="pin_{index}_body" type="capsule" size="0.105 0.20" pos="0 0 0.32" rgba="0.92 0.92 0.94 1"/>
      <geom name="pin_{index}_neck" type="cylinder" size="0.065 0.055" pos="0 0 0.55" rgba="0.92 0.92 0.94 1"/>
      <geom name="pin_{index}_head" type="sphere" size="0.075" pos="0 0 0.64" rgba="0.92 0.92 0.94 1"/>
      <geom name="pin_{index}_stripe" type="cylinder" size="0.068 0.012" pos="0 0 0.555" rgba="0.82 0.05 0.04 1" contype="0" conaffinity="0"/>
    </body>'''


def make_bowling_xml() -> str:
    """Build a standard ten-pin arrangement on a MuJoCo ground plane."""
    positions = []
    spacing = 0.27
    for row in range(4):
        x = 1.25 + row * spacing * 0.86
        for column in range(row + 1):
            positions.append((x, (column - row / 2.0) * spacing))
    pins = "\n".join(_pin_xml(i + 1, x, y) for i, (x, y) in enumerate(positions))
    return f'''<mujoco model="bowling">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81" iterations="80" solver="Newton"/>
  <size njmax="1000" nconmax="500"/>
  <visual><headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25"/><map znear="0.01" zfar="30"/></visual>
  <asset>
    <texture name="floor_texture" type="2d" builtin="checker" width="512" height="512" rgb1="0.72 0.55 0.34" rgb2="0.58 0.40 0.23"/>
    <material name="lane" texture="floor_texture" texrepeat="6 2"/>
  </asset>
  <worldbody>
    <light name="overhead" pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="ground" type="plane" size="12 6 0.1" material="lane" friction="0.7 0.01 0.005"/>
    <geom name="back_wall" type="box" size="0.12 2.5 0.5" pos="3 0 0.5" rgba="0.25 0.25 0.28 1"/>
    <body name="ball" pos="-2 0 0.14">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.14" mass="7" rgba="0.03 0.08 0.65 1" friction="0.5 0.02 0.01"/>
    </body>
    {pins}
  </worldbody>
</mujoco>'''


class BowlingEnv(gym.Env):
    """Force-controlled bowling scene; pins are fully moveable rigid bodies."""
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode: str | None = None, max_steps: int = 1500):
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.model = mujoco.MjModel.from_xml_string(make_bowling_xml())
        self.data = mujoco.MjData(self.model)
        self._ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        self._viewer: Any = None
        self._step_count = 0
        self.action_space = spaces.Box(-80.0, 80.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.model.nq + self.model.nv,), dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        return np.concatenate((self.data.qpos, self.data.qvel)).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (3,):
            raise ValueError(f"Expected action shape (3,), got {action.shape}")
        self.data.xfrc_applied[self._ball_id, :3] = np.clip(action, -80.0, 80.0)
        mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[self._ball_id, :] = 0
        self._step_count += 1
        reward = -0.001 * float(np.dot(action, action))
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, False, self._step_count >= self.max_steps, {}

    def render(self):
        if self.render_mode == "rgb_array":
            if self._viewer is None:
                self._viewer = mujoco.Renderer(self.model, height=480, width=640)
            self._viewer.update_scene(self.data)
            return self._viewer.render()
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None and self.render_mode == "human":
            self._viewer.close()
        self._viewer = None


if __name__ == "__main__":
    env = BowlingEnv(render_mode="human")
    try:
        env.reset()
        # The passive viewer runs on another thread; step at wall-clock speed
        # and wait for the user to close the window before tearing it down.
        while env._viewer is None or env._viewer.is_running():
            env.step(np.array([35.0, 0.0, 0.0], dtype=np.float32))
            time.sleep(1.0 / 60.0)
    finally:
        env.close()
