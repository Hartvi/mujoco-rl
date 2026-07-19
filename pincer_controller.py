"""MuJoCo model and controller for the two-cube pincer."""
from __future__ import annotations

import numpy as np
import mujoco


class PincerController:
    """Controller for the cube pair pose and actuated distance."""
    DISTANCE_JOINT = "cube_distance"
    DISTANCE_ACTUATOR = "distance_command"
    POSITION_KP = 500.0
    POSITION_KD = 40.0
    MAX_FORCE = 100.0
    ORIENTATION_KP = 1.0
    ORIENTATION_KD = 0.1
    MAX_TORQUE = 0.2
    MAX_LINEAR_SPEED = 0.5  # metres per simulated second
    MAX_ANGULAR_SPEED = 0.5  # radians per simulated second
    MAX_DISTANCE_SPEED = 0.05  # metres per simulated second
    OBJECT_BODY = "cube_pair"

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, control_dt: float = 0.02):
        self.model = model
        self.data = data
        self.control_dt = control_dt
        self.max_translation_delta = self.MAX_LINEAR_SPEED * self.control_dt
        self.max_rotation_delta = self.MAX_ANGULAR_SPEED * self.control_dt
        self.max_distance_delta = self.MAX_DISTANCE_SPEED * self.control_dt
        self.body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, self.OBJECT_BODY)
        self.object_joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, "object_pose")
        self.object_qpos_id = int(model.jnt_qposadr[self.object_joint_id])
        self.object_dof_id = int(model.jnt_dofadr[self.object_joint_id])
        self.joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, self.DISTANCE_JOINT)
        self.actuator_id = self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, self.DISTANCE_ACTUATOR)
        self.qpos_id = int(model.jnt_qposadr[self.joint_id])
        self.distance_range = model.jnt_range[self.joint_id].copy()
        self.target_distance = float(data.qpos[self.qpos_id])
        self.target_position = data.qpos[self.object_qpos_id:self.object_qpos_id + 3].copy()
        self.target_quat = data.qpos[self.object_qpos_id + 3:self.object_qpos_id + 7].copy()
        self._apply_pose_control()

    def _name_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Pincer XML is missing {name!r}")
        return object_id

    def reset(self) -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "touching")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.target_distance = float(self.data.qpos[self.qpos_id])
        self.target_position = self.data.qpos[self.object_qpos_id:self.object_qpos_id + 3].copy()
        self.target_quat = self.data.qpos[self.object_qpos_id + 3:self.object_qpos_id + 7].copy()
        self._apply_pose_control()
        self.data.ctrl[self.actuator_id] = self.target_distance
        mujoco.mj_forward(self.model, self.data)

    def observation(self) -> np.ndarray:
        position = self.data.xpos[self.body_id].copy()
        orientation = np.zeros(3, dtype=np.float64)
        mujoco.mju_quat2Vel(orientation, self.data.xquat[self.body_id], 1.0)
        return np.concatenate((position, orientation, [self.data.qpos[self.qpos_id]])).astype(np.float32)

    def apply_delta(self, action: np.ndarray) -> None:
        """Apply [dx, dy, dz, dRx, dRy, dRz, d_distance] deltas."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != 7:
            raise ValueError(f"Pincer action must contain 7 pose values, not {action.size}")
        self.target_position += np.clip(action[:3], -self.max_translation_delta, self.max_translation_delta)
        self.target_quat = self._quat_mul(self.target_quat, self._rotvec_to_quat(np.clip(action[3:6], -self.max_rotation_delta, self.max_rotation_delta)))
        self._apply_pose_control()
        self.target_distance = float(np.clip(self.target_distance + np.clip(action[6], -self.max_distance_delta, self.max_distance_delta), *self.distance_range))
        self.data.ctrl[self.actuator_id] = self.target_distance

    def hold_pose(self) -> None:
        """Keep actuators aimed at the command without overriding contact physics."""
        self._apply_pose_control()
        self.data.ctrl[self.actuator_id] = self.target_distance

    def _apply_pose_control(self) -> None:
        velocity = self.data.qvel[self.object_dof_id:self.object_dof_id + 6]
        position_error = self.target_position - self.data.xpos[self.body_id]
        force = self.POSITION_KP * position_error - self.POSITION_KD * velocity[:3]

        actual_quat = self.data.xquat[self.body_id]
        quat_error = self._quat_mul(
            self.target_quat,
            np.array([actual_quat[0], *(-actual_quat[1:])]),
        )
        if quat_error[0] < 0.0:
            quat_error *= -1.0
        rotation_error = np.zeros(3, dtype=np.float64)
        mujoco.mju_quat2Vel(rotation_error, quat_error, 1.0)
        torque = (
            self.ORIENTATION_KP * rotation_error
            - self.ORIENTATION_KD * velocity[3:]
        )

        self.data.xfrc_applied[self.body_id, :3] = self._clip_norm(
            force, self.MAX_FORCE
        )
        self.data.xfrc_applied[self.body_id, 3:] = self._clip_norm(
            torque, self.MAX_TORQUE
        )

    @staticmethod
    def _clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector if norm <= maximum else vector * (maximum / norm)

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw
        ])

    @staticmethod
    def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return np.concatenate(([np.cos(angle / 2.0)], np.sin(angle / 2.0) * rotvec / angle))
