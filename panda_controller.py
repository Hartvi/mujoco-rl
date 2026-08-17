"""Panda end-effector controller used by the bowling environment."""

from __future__ import annotations

from typing import Any, ClassVar

import mujoco
import numpy as np


class PandaController:
    ARM_JOINTS: ClassVar[list[str]] = [f"panda_joint{i}" for i in range(1, 8)]
    RESET_Q: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.array(
        [0.0, 0.65, 0.0, -1.5, 0.0, 1.0, 0.5], dtype=np.float64
    )

    def __init__(
        self, model: mujoco.MjModel, data: mujoco.MjData, control_dt: float = 0.02
    ) -> None:
        self.model: mujoco.MjModel = model
        self.data: mujoco.MjData = data
        self.site_id: int = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "panda_hand"
        )
        self.joint_ids: list[int] = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in self.ARM_JOINTS
        ]
        self.qpos_ids: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.array(
            [model.jnt_qposadr[i] for i in self.joint_ids]
        )
        self.actuator_ids: np.ndarray[tuple[Any, ...], np.dtype[Any]] = np.array(
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"panda_motor{i}"
                )
                for i in range(1, 8)
            ]
        )
        self.target_rpy: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(
            3, dtype=np.float64
        )
        self.target_pos: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(
            3, dtype=np.float64
        )
        self.target_quat: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float64
        )
        self.control_dt: float = control_dt

    def reset(self) -> None:
        self.data.qpos[self.qpos_ids] = self.RESET_Q
        self.data.ctrl[self.actuator_ids] = self.RESET_Q
        mujoco.mj_forward(self.model, self.data)
        self.target_pos = self.data.site_xpos[self.site_id].copy()
        self.target_quat = self._site_quat()

    def observation(self) -> np.ndarray:
        pos = self.data.site_xpos[self.site_id].copy()
        quat: np.ndarray[tuple[Any, ...], np.dtype[Any]] = self._site_quat()
        orientation: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(3)
        mujoco.mju_quat2Vel(orientation, quat, 1.0)
        return np.concatenate((pos, orientation)).astype(np.float32)

    def apply_delta(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64)
        self.target_pos = self.target_pos + np.clip(action[:3], -0.05, 0.05)
        self.target_pos = np.clip(self.target_pos, [-2.8, -1.0, 0.08], [-1.5, 1.0, 1.4])
        self.target_quat = self._quat_mul(
            self.target_quat,
            self._rotvec_to_quat(np.clip(action[3:], -0.2, 0.2)),
        )

        jacp: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
            (3, self.model.nv)
        )
        jacr: np.ndarray[tuple[int, int], np.dtype[np.float64]] = np.zeros(
            (3, self.model.nv)
        )
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        error = self.target_pos - self.data.site_xpos[self.site_id]
        current_quat: np.ndarray[tuple[Any, ...], np.dtype[Any]] = self._site_quat()
        orientation_error: np.ndarray[tuple[Any, ...], np.dtype[Any]] = self._quat_mul(
            self.target_quat, self._quat_conjugate(current_quat)
        )
        if orientation_error[0] < 0.0:
            orientation_error *= -1.0
        error = np.concatenate((error, 2.0 * orientation_error[1:]))
        jac: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = np.vstack(
            (jacp[:, self._vel_indices()], jacr[:, self._vel_indices()])
        )
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), error)
        q: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = self.data.qpos[
            self.qpos_ids
        ].copy()
        q_target = q + np.clip(dq, -0.08, 0.08)
        self.data.ctrl[self.actuator_ids] = q_target

    def _site_quat(self) -> np.ndarray:
        quat: np.ndarray[tuple[int], np.dtype[np.float64]] = np.zeros(
            4, dtype=np.float64
        )
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site_id])
        return quat

    @staticmethod
    def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)

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
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-9:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        axis = rotvec / angle
        half: float = 0.5 * angle
        return np.concatenate(([np.cos(half)], np.sin(half) * axis))

    def _vel_indices(self) -> np.ndarray:
        return np.array([self.model.jnt_dofadr[i] for i in self.joint_ids])
