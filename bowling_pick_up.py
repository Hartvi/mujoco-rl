"""Simple Gymnasium bowling environment driven by the XML pincer."""

from __future__ import annotations

import time
from typing import Any, SupportsFloat, TypeAlias

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

from bowling_scene import make_bowling_xml, PinComponent
from pincer_controller import PincerController


ObsType: TypeAlias = dict[str, np.ndarray]


class BowlingPickUp(gym.Env):
    action_space: spaces.Box
    observation_space: spaces.Dict
    metadata: dict[str, Any] = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    FRAME_SKIP = 50
    PINCER_STATE_SIZE = 8
    PIN_STATE_SIZE = 14
    MAX_PINS = 10
    SUCCESS_REWARD = 0.0
    START_RADIUS = 0.5
    START_PIN_CLEARANCE = 0.15
    STRIKE_POINT_HEIGHT = 0.25
    # reward modifier
    AWAY_DISTANCE_MULTIPLIER = 0.0
    # rewards
    MIN_GROUND_CLEARANCE = 0.01
    NEWLY_STOOD_UP_REWARD = 0.0
    PIN_TOUCH_REWARD = 0.0
    GROUND_PENALTY_SCALE = 0.1
    DISTANCE_SCALE = 0.0
    ACTION_PENALTY_SCALE = 0.0
    BETWEEN_PIN_SCALE = 0.1
    TIME_PENALTY = 0.002

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 500,
        num_pins: int = 10,
        pin_component: PinComponent | None = PinComponent.HEAD,
        pins_fallen: bool = False,
    ) -> None:
        if render_mode not in self.metadata["render_modes"] + [None]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if not 1 <= num_pins <= self.MAX_PINS:
            raise ValueError(f"num_pins must be between 1 and {self.MAX_PINS}")
        self.render_mode = render_mode
        self.max_steps: int = max_steps
        self.num_pins: int = num_pins
        self.bowling_scene: mujoco.MjModel = mujoco.MjModel.from_xml_string(
            make_bowling_xml(
                include_pincer=True,
                pins_fallen=pins_fallen,
                num_pins=num_pins,
            )
        )
        self.part: PinComponent | None = pin_component
        self.data = mujoco.MjData(self.bowling_scene)
        self._viewer: Any = None
        self._renderer: mujoco.Renderer | None = None
        self._step_count = 0
        self._previous_fallen = 0
        self._rewarded_picked_up_pins: set[int] = set()
        self._target_pin_id: int | None = None
        self._touched_pins: set[int] = set()
        self._previous_pin_distance = 0.0
        self._last_action_time = 0.0
        self._last_action_dt = 0.0
        self.task = "pick up the pins"
        self.control_dt = float(self.bowling_scene.opt.timestep * self.FRAME_SKIP)
        self._ee = PincerController(
            self.bowling_scene, self.data, control_dt=self.control_dt
        )
        self._pin_ids: list[int] = [
            mujoco.mj_name2id(self.bowling_scene, mujoco.mjtObj.mjOBJ_BODY, f"pin_{i}")
            for i in range(1, num_pins + 1)
        ]
        self._cube_geom_ids: list[int] = [
            mujoco.mj_name2id(self.bowling_scene, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("cube_1", "cube_2")
        ]

        self._pin_component_ids: dict[PinComponent, list[int]] = (
            self.get_pin_component_ids()
        )
        # Policy action layout: [vx, vy, vz, wx, wy, wz, jaw_velocity].
        # Each component is normalized to [-1, 1] and converted to a pose
        # delta using the controller's physical rate limits and control_dt.
        self.action_space = spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self._action_delta_scale: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = (
            np.array(
                [self._ee.max_translation_delta] * 3
                + [self._ee.max_rotation_delta] * 3
                + [self._ee.max_distance_delta],
                dtype=np.float64,
            )
        )
        observation_size: int = self.PINCER_STATE_SIZE + self.PIN_STATE_SIZE * num_pins
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(
                    -np.inf, np.inf, shape=(observation_size,), dtype=np.float32
                ),
            }
        )

    def get_pin_component_ids(self) -> dict[PinComponent, list[int]]:
        _pin_component_ids: dict[PinComponent, list[int]] = {}
        for pc in PinComponent.__members__.values():
            _pin_component_ids[pc] = [
                mujoco.mj_name2id(
                    self.bowling_scene,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"pin_{i}_{pc}",
                )
                for i in range(1, self.num_pins + 1)
            ]
        self.pin2component_id: dict[PinComponent, dict[int, int]] = {
            p: dict(zip(self._pin_ids, _pin_component_ids[p]))
            for p in PinComponent.__members__.values()
        }
        return _pin_component_ids

    def _fallen_pins(self) -> int:
        return len(self._fallen_pin_ids())

    def _pincer_ground_clearance(self) -> float:
        clearances = []
        for geom_id in self._cube_geom_ids:
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            vertical_half_extent = float(
                np.abs(rotation[2]) @ self.bowling_scene.geom_size[geom_id]
            )
            clearances.append(
                float(self.data.geom_xpos[geom_id, 2]) - vertical_half_extent
            )
        return min(clearances)

    def both_touching_pins(self) -> bool:
        if self.part is None or self._target_pin_id is None:
            return bool(self._touching_pins())
        pin_part_geom_id = self.pin2component_id[self.part][self._target_pin_id]
        cube_geom_ids: set[int] = set(self._cube_geom_ids)
        # geom1 is the cube1, geom2 is the cube2 or vice versa
        # AND both are touching the same pin
        for contact in self.data.contact:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            # both cubes touching the same pin
            if geom1 in cube_geom_ids and geom1 == pin_part_geom_id:
                cube_geom_ids.remove(geom1)
            if geom2 in cube_geom_ids and geom2 == pin_part_geom_id:
                cube_geom_ids.remove(geom2)
        return not bool(cube_geom_ids)

    def pin_between_cubes(self) -> float:
        if self.part is None or self._target_pin_id is None:
            return 0.0
        pin_part_geom_id = self.pin2component_id[self.part][self._target_pin_id]
        cube1_pos = self.data.geom_xpos[self._cube_geom_ids[0]]
        cube2_pos = self.data.geom_xpos[self._cube_geom_ids[1]]
        pin_pos = self.data.geom_xpos[pin_part_geom_id]
        cube1_to_cube2 = cube2_pos - cube1_pos
        cube1_to_pin = pin_pos - cube1_pos
        # project the pin position onto the line between the cubes
        projection_length = np.dot(cube1_to_pin, cube1_to_cube2) / np.linalg.norm(
            cube1_to_cube2
        )
        projection_point = (
            cube1_pos
            + (projection_length / np.linalg.norm(cube1_to_cube2)) * cube1_to_cube2
        )

        # distance from the pin to the line between the cubes
        # double linear when outside the pincer, quite small when between pincers
        # the reward includes the correct rotation
        distance = 0.5 * np.linalg.norm(projection_point - cube1_pos - cube2_pos)
        return float(distance)

    def _touching_pins(self) -> set[int]:
        touching = set()
        cube_geom_ids: set[int] = set(self._cube_geom_ids)
        pin_body_ids: set[int] = set(self._pin_ids)
        for contact in self.data.contact:
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in cube_geom_ids:
                other_geom: int = geom2
            elif geom2 in cube_geom_ids:
                other_geom = geom1
            else:
                continue
            pin_body_id = int(self.bowling_scene.geom_bodyid[other_geom])
            if pin_body_id in pin_body_ids:
                touching.add(pin_body_id)
        return touching

    def _fallen_pin_ids(self) -> set[int]:
        return {
            body_id
            for body_id in self._pin_ids
            if float(self.data.xmat[body_id].reshape(3, 3)[2, 2]) < 0.75
        }

    def _strike_point(self, body_id: int | None) -> np.ndarray:
        offset = np.array([0.0, 0.0, self.STRIKE_POINT_HEIGHT])
        if body_id is None:
            return offset
        return self.data.xpos[body_id] + offset

    def _select_target_pin(self, fallen_pin_ids: set[int]) -> None:
        pincer_position = self.data.xpos[self._ee.body_id]
        candidates: list[int] = [
            body_id for body_id in self._pin_ids if body_id in fallen_pin_ids
        ]
        self._target_pin_id = min(
            candidates,
            key=lambda body_id: np.linalg.norm(
                self._strike_point(body_id) - pincer_position
            ),
            default=None,
        )

    def _relevant_pin_distance(self) -> float:
        if self._target_pin_id is None:
            return 0.0
        if self.part is None:
            return float(
                np.linalg.norm(
                    self._strike_point(self._target_pin_id)
                    - self.data.xpos[self._ee.body_id]
                )
            )
        else:
            return float(
                np.linalg.norm(
                    self.data.geom_xpos[
                        self.pin2component_id[self.part][self._target_pin_id]
                    ]
                    - self.data.xpos[self._ee.body_id]
                )
            )

    def _random_pincer_start_xy(self) -> np.ndarray:
        pin_positions: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.asarray(
            [self.data.xpos[body_id, :2] for body_id in self._pin_ids]
        )
        rack_center = pin_positions.mean(axis=0)
        best_candidate = rack_center.copy()
        best_clearance: float = -np.inf
        for _ in range(100):
            radius = self.START_RADIUS * np.sqrt(self.np_random.random())
            angle: float = self.np_random.uniform(0.0, 2.0 * np.pi)
            candidate = rack_center + radius * np.array([np.cos(angle), np.sin(angle)])
            clearance = float(np.min(np.linalg.norm(pin_positions - candidate, axis=1)))
            if clearance >= self.START_PIN_CLEARANCE:
                return candidate
            if clearance > best_clearance:
                best_candidate = candidate
                best_clearance = clearance
        return best_candidate

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        reset:
          set pins to knocked over state
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.bowling_scene, self.data)
        self.data.qpos[:] = self.bowling_scene.qpos0
        self.data.qvel[:] = 0.0
        self._ee.reset()
        self.data.qpos[self._ee.object_qpos_id : self._ee.object_qpos_id + 2] = (
            self._random_pincer_start_xy()
        )
        self._ee.sync_target_to_pose()
        mujoco.mj_forward(self.bowling_scene, self.data)
        self._step_count = 0
        self._previous_fallen = self.num_pins
        self._rewarded_picked_up_pins.clear()
        self._select_target_pin(set())
        self._touched_pins.clear()
        self._previous_pin_distance = self._relevant_pin_distance()
        self._last_action_time = float(self.data.time)
        self._last_action_dt = 0.0
        return self._get_observation(), {
            "fallen_pins": self._previous_fallen,
            "task": self.task,
        }

    def _body_velocity(self, body_id: int) -> np.ndarray:
        velocity: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(
            6, dtype=np.float64
        )
        mujoco.mj_objectVelocity(
            self.bowling_scene,
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
        return np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ]
        )

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
        relative_quat: np.ndarray[tuple[Any, ...], np.dtype[Any]] = self._quat_mul(
            self._quat_conjugate(pincer_quat),
            self.data.xquat[body_id],
        )
        if relative_quat[0] < 0.0:
            relative_quat *= -1.0

        pin_velocity: np.ndarray[tuple[Any, ...], np.dtype[Any]] = self._body_velocity(
            body_id
        )
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
        return np.concatenate(
            (
                relative_position,
                relative_quat,
                [fallen],
                relative_linear_velocity,
                relative_angular_velocity,
            )
        )

    def _get_observation(self) -> ObsType:
        pincer_state: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.concatenate(
            (
                self._ee.observation(),
                [self._last_action_dt],
            )
        )
        pincer_position = self.data.xpos[self._ee.body_id]
        pincer_rotation = self.data.xmat[self._ee.body_id].reshape(3, 3)
        world_to_pincer = pincer_rotation.T
        pincer_quat = self.data.xquat[self._ee.body_id]
        pincer_velocity: np.ndarray[tuple[Any, ...], np.dtype[Any]] = (
            self._body_velocity(self._ee.body_id)
        )
        pin_states: list[np.ndarray[tuple[Any, ...], np.dtype[Any]]] = [
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
            "observation.state": np.concatenate((pincer_state, *pin_states)).astype(
                np.float32
            )
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        if not self.action_space.contains(action.astype(np.float32)):
            raise ValueError(f"Action outside space: {action}")
        physical_action = action * self._action_delta_scale
        action_time = float(self.data.time)
        self._ee.apply_delta(physical_action)
        for _ in range(self.FRAME_SKIP):
            self._ee.hold_pose()
            mujoco.mj_step(self.bowling_scene, self.data)
        self._last_action_dt = float(self.data.time - action_time)
        self._last_action_time = float(self.data.time)
        self._step_count += 1
        fallen_pin_ids: set[int] = self._fallen_pin_ids()
        fallen: int = len(fallen_pin_ids)
        terminated: bool = fallen == 0
        truncated: bool = self._step_count >= self.max_steps

        # Reward for standing up
        newly_stood_up: int = max(0, self._previous_fallen - fallen)
        stood_up_reward: float = self.NEWLY_STOOD_UP_REWARD * float(newly_stood_up)
        self._previous_fallen = fallen

        # Reward for approach
        pin_distance = self._relevant_pin_distance()
        distance_progress: float = self._previous_pin_distance - pin_distance
        distance_multiplier: float = (
            self.AWAY_DISTANCE_MULTIPLIER if distance_progress < 0.0 else 1.0
        )
        distance_reward = self.DISTANCE_SCALE * distance_multiplier * distance_progress
        self._previous_pin_distance = pin_distance

        # ground clearance
        ground_clearance: float = self._pincer_ground_clearance()
        ground_reward: float = -self.GROUND_PENALTY_SCALE * max(
            0.0, self.MIN_GROUND_CLEARANCE - ground_clearance
        )
        rotation_reward = 0.0
        action_reward: float = -self.ACTION_PENALTY_SCALE * float(
            np.dot(action, action)
        )

        # reward for grasping
        # pin between the cubes, but not touching the pin (2D projection)
        between_reward: float = -self.BETWEEN_PIN_SCALE * self.pin_between_cubes()
        touching_reward: float = self.PIN_TOUCH_REWARD * float(
            self.both_touching_pins()
        )
        # touching_pins: set[int] = self._touching_pins()
        # newly_touched_pins: set[int] = touching_pins - self._touched_pins
        # self._touched_pins.update(touching_pins)
        # touch_reward: float = self.PIN_TOUCH_REWARD * float(len(newly_touched_pins))

        success_reward: float = self.SUCCESS_REWARD if terminated else 0.0
        open_close_reward = 0.0
        time_reward: float = -self.TIME_PENALTY
        reward: float = (
            distance_reward
            + stood_up_reward
            + open_close_reward
            + rotation_reward
            + ground_reward
            + action_reward
            + between_reward
            + touching_reward
            + success_reward
            + time_reward
        )
        if self.render_mode == "human":
            self.render()
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            {
                "fallen_pins": fallen,
                "success": terminated,
                "is_success": terminated,
                "task": self.task,
                "reward.distance": distance_reward,
                "reward.rotation": rotation_reward,
                "reward.ground_clearance": ground_reward,
                "reward.fallen_pins": stood_up_reward,
                "reward.open_close": open_close_reward,
                "reward.action": action_reward,
                "reward.pin_touch": touching_reward,
                "reward.pin_between": between_reward,
                "reward.success": success_reward,
                "reward.time": time_reward,
                "distance.relevant_pin": pin_distance,
                "distance.ground_clearance": ground_clearance,
                "pin.part": self.part.name if self.part is not None else None,
            },
        )

    def _render_camera(self, camera_name: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.bowling_scene, height=480, width=640)
        self._renderer.update_scene(self.data, camera=camera_name)
        return self._renderer.render().copy()

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_camera("laptop_camera")
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.bowling_scene, self.data
                )
            self._viewer.sync()
        return None

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
        if self._renderer is not None:
            self._renderer.close()
        self._viewer = None
        self._renderer = None


if __name__ == "__main__":
    env = BowlingPickUp(render_mode="human")
    try:
        env.reset()
        while env._viewer is None or env._viewer.is_running():
            env.step(np.zeros(7, dtype=np.float32))
            time.sleep(1.0 / 60.0)
    finally:
        env.close()
