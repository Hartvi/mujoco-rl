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
    FRAME_SKIP = 10
    PINCER_STATE_SIZE = 8
    PIN_STATE_SIZE = 14
    MAX_PINS = 10
    NEWLY_FALLEN_REWARD = 50.0
    PIN_TOUCH_REWARD = 5.0
    SUCCESS_REWARD = 100.0
    START_RADIUS = 0.5
    START_PIN_CLEARANCE = 0.15
    GROUND_CLEARANCE_TARGET = 0.2
    GROUND_CLEARANCE_REWARD_SCALE = 0.1
    DISTANCE_SCALE = 5.0
    AWAY_DISTANCE_MULTIPLIER = 1.25
    ACTION_PENALTY_SCALE = 0.01
    JAW_MOTION_PENALTY_SCALE = 0.005
    JAW_CLOSE_REWARD_SCALE = 0.02
    JAW_CLOSE_DISTANCE = 0.15

    def __init__(self, render_mode: str | None = None, max_steps: int = 500, num_pins: int = 10):
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if not 1 <= num_pins <= self.MAX_PINS:
            raise ValueError(f"num_pins must be between 1 and {self.MAX_PINS}")
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.num_pins = num_pins
        self.model = mujoco.MjModel.from_xml_string(make_bowling_xml(include_pincer=True))
        self.data = mujoco.MjData(self.model)
        self._viewer: Any = None
        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self._previous_fallen = 0
        self._touched_pins: set[int] = set()
        self._previous_pin_distance = 0.0
        self._last_action_time = 0.0
        self._last_action_dt = 0.0
        self.task = "bowl the pins with the pincer"
        self.control_dt = float(self.model.opt.timestep * self.FRAME_SKIP)
        self._ee = PincerController(self.model, self.data, control_dt=self.control_dt)
        self._pin_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"pin_{i}")
            for i in range(1, num_pins + 1)
        ]
        self._cube_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("cube_1", "cube_2")
        ]

        # Policy action layout: [vx, vy, vz, wx, wy, wz, jaw_velocity].
        # Each component is normalized to [-1, 1] and converted to a pose
        # delta using the controller's physical rate limits and control_dt.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self._action_delta_scale = np.array(
            [self._ee.max_translation_delta] * 3
            + [self._ee.max_rotation_delta] * 3
            + [self._ee.max_distance_delta],
            dtype=np.float64,
        )
        observation_size = self.PINCER_STATE_SIZE + self.PIN_STATE_SIZE * num_pins
        self.observation_space = spaces.Dict({
            "observation.state": spaces.Box(
                -np.inf, np.inf, shape=(observation_size,), dtype=np.float32
            ),
        })

    def _fallen_pins(self) -> int:
        return sum(
            float(self.data.xmat[body_id].reshape(3, 3)[2, 2]) < 0.75
            for body_id in self._pin_ids
        )

    def _pincer_ground_clearance(self) -> float:
        clearances = []
        for geom_id in self._cube_geom_ids:
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            vertical_half_extent = float(
                np.abs(rotation[2]) @ self.model.geom_size[geom_id]
            )
            clearances.append(
                float(self.data.geom_xpos[geom_id, 2]) - vertical_half_extent
            )
        return min(clearances)

    def _touching_pins(self) -> set[int]:
        touching = set()
        cube_geom_ids = set(self._cube_geom_ids)
        pin_body_ids = set(self._pin_ids)
        for contact in self.data.contact:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in cube_geom_ids:
                other_geom = geom2
            elif geom2 in cube_geom_ids:
                other_geom = geom1
            else:
                continue
            pin_body_id = int(self.model.geom_bodyid[other_geom])
            if pin_body_id in pin_body_ids:
                touching.add(pin_body_id)
        return touching

    def _relevant_pin_distance(self) -> float:
        pincer_position = self.data.xpos[self._ee.body_id]
        pin_positions = np.asarray([self.data.xpos[body_id] for body_id in self._pin_ids])
        fallen_mask = np.asarray([
            self.data.xmat[body_id].reshape(3, 3)[2, 2] < 0.75
            for body_id in self._pin_ids
        ])
        candidates = pin_positions[~fallen_mask]
        if candidates.size == 0:
            return 0.0
        return float(np.min(np.linalg.norm(candidates - pincer_position, axis=1)))

    def _random_pincer_start_xy(self) -> np.ndarray:
        pin_positions = np.asarray([
            self.data.xpos[body_id, :2] for body_id in self._pin_ids
        ])
        rack_center = pin_positions.mean(axis=0)
        best_candidate = rack_center.copy()
        best_clearance = -np.inf
        for _ in range(100):
            radius = self.START_RADIUS * np.sqrt(self.np_random.random())
            angle = self.np_random.uniform(0.0, 2.0 * np.pi)
            candidate = rack_center + radius * np.array([
                np.cos(angle), np.sin(angle)
            ])
            clearance = float(np.min(
                np.linalg.norm(pin_positions - candidate, axis=1)
            ))
            if clearance >= self.START_PIN_CLEARANCE:
                return candidate
            if clearance > best_clearance:
                best_candidate = candidate
                best_clearance = clearance
        return best_candidate

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self._ee.reset()
        self.data.qpos[
            self._ee.object_qpos_id:self._ee.object_qpos_id + 2
        ] = self._random_pincer_start_xy()
        self._ee.sync_target_to_pose()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        self._previous_fallen = 0
        self._touched_pins.clear()
        self._previous_pin_distance = self._relevant_pin_distance()
        self._last_action_time = float(self.data.time)
        self._last_action_dt = 0.0
        return self._get_observation(), {"fallen_pins": 0, "task": self.task}

    def _body_velocity(self, body_id: int) -> np.ndarray:
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        return velocity

    @staticmethod
    def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]])

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ])

    def _pin_observation(
        self,
        body_id: int,
        pincer_position: np.ndarray,
        world_to_pincer: np.ndarray,
        pincer_quat: np.ndarray,
        pincer_velocity: np.ndarray,
    ) -> np.ndarray:
        relative_position = world_to_pincer @ (
            self.data.xpos[body_id] - pincer_position
        )
        relative_quat = self._quat_mul(
            self._quat_conjugate(pincer_quat),
            self.data.xquat[body_id],
        )
        if relative_quat[0] < 0.0:
            relative_quat *= -1.0

        pin_velocity = self._body_velocity(body_id)
        relative_angular_velocity = world_to_pincer @ (
            pin_velocity[:3] - pincer_velocity[:3]
        )
        relative_linear_velocity = world_to_pincer @ (
            pin_velocity[3:]
            - pincer_velocity[3:]
            - np.cross(
                pincer_velocity[:3],
                self.data.xpos[body_id] - pincer_position,
            )
        )
        fallen = float(self.data.xmat[body_id].reshape(3, 3)[2, 2] < 0.75)

        # Per-pin layout: relative position (3), relative quaternion (4),
        # fallen flag (1), relative linear velocity (3), relative angular
        # velocity (3).
        return np.concatenate((
            relative_position,
            relative_quat,
            [fallen],
            relative_linear_velocity,
            relative_angular_velocity,
        ))

    def _get_observation(self) -> dict[str, np.ndarray]:
        pincer_state = np.concatenate((
            self._ee.observation(),
            [self._last_action_dt],
        ))
        pincer_position = self.data.xpos[self._ee.body_id]
        pincer_rotation = self.data.xmat[self._ee.body_id].reshape(3, 3)
        world_to_pincer = pincer_rotation.T
        pincer_quat = self.data.xquat[self._ee.body_id]
        pincer_velocity = self._body_velocity(self._ee.body_id)
        pin_states = [
            self._pin_observation(
                body_id,
                pincer_position,
                world_to_pincer,
                pincer_quat,
                pincer_velocity,
            )
            for body_id in self._pin_ids
        ]
        return {
            "observation.state": np.concatenate(
                (pincer_state, *pin_states)
            ).astype(np.float32)
        }

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(action.astype(np.float32)):
            raise ValueError(f"Action outside space: {action}")
        physical_action = action * self._action_delta_scale
        action_time = float(self.data.time)
        self._ee.apply_delta(physical_action)
        for _ in range(self.FRAME_SKIP):
            self._ee.hold_pose()
            mujoco.mj_step(self.model, self.data)
        self._last_action_dt = float(self.data.time - action_time)
        self._last_action_time = float(self.data.time)
        self._step_count += 1
        fallen = self._fallen_pins()
        terminated = fallen == self.num_pins
        truncated = self._step_count >= self.max_steps
        pin_distance = self._relevant_pin_distance()
        distance_progress = self._previous_pin_distance - pin_distance
        distance_multiplier = (
            self.AWAY_DISTANCE_MULTIPLIER if distance_progress < 0.0 else 1.0
        )
        distance_reward = (
            self.DISTANCE_SCALE * distance_multiplier * distance_progress
        )
        self._previous_pin_distance = pin_distance
        ground_clearance = self._pincer_ground_clearance()
        ground_reward = - self.GROUND_CLEARANCE_REWARD_SCALE * float(np.clip(
            np.abs(ground_clearance - self.GROUND_CLEARANCE_TARGET), -1.0, 1.0
        ))
        # Rewards must be scalar. Penalize rotation magnitude so clockwise and
        # counter-clockwise commands are treated equally.
        rotation_reward = -float(np.linalg.norm(physical_action[3:6]))
        action_reward = -self.ACTION_PENALTY_SCALE * float(
            np.dot(action, action)
        )
        newly_fallen = max(0, fallen - self._previous_fallen)
        fallen_reward = self.NEWLY_FALLEN_REWARD * float(newly_fallen)
        self._previous_fallen = fallen
        touching_pins = self._touching_pins()
        newly_touched_pins = touching_pins - self._touched_pins
        self._touched_pins.update(touching_pins)
        touch_reward = self.PIN_TOUCH_REWARD * float(len(newly_touched_pins))
        success_reward = self.SUCCESS_REWARD if terminated else 0.0
        jaw_command = float(action[6])
        close_proximity = max(
            0.0, 1.0 - pin_distance / self.JAW_CLOSE_DISTANCE
        )
        open_close_reward = (
            -self.JAW_MOTION_PENALTY_SCALE * abs(jaw_command)
            + self.JAW_CLOSE_REWARD_SCALE
            * max(0.0, -jaw_command)
            * close_proximity
        )
        reward = (distance_reward + fallen_reward + open_close_reward
                  + rotation_reward + ground_reward + action_reward
                  + touch_reward + success_reward)
        if self.render_mode == "human":
            self.render()
        return self._get_observation(), reward, terminated, truncated, {
            "fallen_pins": fallen,
            "newly_fallen": newly_fallen,
            "newly_touched_pins": len(newly_touched_pins),
            "success": terminated,
            "is_success": terminated,
            "task": self.task,
            "reward.distance": distance_reward,
            "reward.rotation": rotation_reward,
            "reward.ground_clearance": ground_reward,
            "reward.fallen_pins": fallen_reward,
            "reward.open_close": open_close_reward,
            "reward.action": action_reward,
            "reward.pin_touch": touch_reward,
            "reward.success": success_reward,
            "distance.relevant_pin": pin_distance,
            "distance.ground_clearance": ground_clearance,
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
