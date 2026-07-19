"""MuJoCo model and controller for the two-cube pincer."""
from __future__ import annotations

import numpy as np
import mujoco


def _pincer_xml(index: int = 0, x: float = 0.0, y: float = 0.0) -> str:
    return f'''
<mujoco model="two_cubes_variable_distance">
  <compiler angle="radian" autolimits="true"/>

  <!-- Disable gravity so the free object remains where it is placed. -->
  <option timestep="0.002" gravity="0 0 0"/>

  <default>
    <!-- size contains half-extents, so 0.01 produces a 2 cm cube. -->
    <geom type="box"
          size="0.01 0.01 0.01"
          density="1000"
          condim="3"/>

    <joint damping="0.2"/>
  </default>

  <worldbody>
    <!-- The complete two-cube object has a free 6-DoF pose. -->
    <body name="cube_pair" pos="0 0 0.05">
      <freejoint name="object_pose"/>

      <geom name="cube_1"
            pos="0 0 0"
            rgba="0.2 0.5 0.9 1"/>

      <!--
        cube_2 starts 2 cm from cube_1.

        Because:
          body position = 0.02 m
          joint reference = 0.02 m

        the joint qpos itself is equal to the centre-to-centre distance.
      -->
      <body name="cube_2_body" pos="0.02 0 0">
        <joint name="cube_distance"
               type="slide"
               axis="1 0 0"
               limited="true"
               range="0.02 0.12"
               ref="0.02"/>

        <geom name="cube_2"
              pos="0 0 0"
              rgba="0.9 0.3 0.2 1"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <!-- ctrl is the desired centre-to-centre distance in metres. -->
    <position name="distance_command"
              joint="cube_distance"
              kp="100"
              ctrllimited="true"
              ctrlrange="0.02 0.12"
              forcelimited="true"
              forcerange="-20 20"/>
  </actuator>

  <keyframe>
    <!--
      qpos:
        root position:       0 0 0.05
        root quaternion:     1 0 0 0
        cube distance:       0.02
    -->
    <key name="touching"
         qpos="0 0 0.05 1 0 0 0 0.02"/>
  </keyframe>
</mujoco>
'''

class PincerController:
    """Controller for the cube pair pose and actuated distance."""
    DISTANCE_JOINT = "cube_distance"
    DISTANCE_ACTUATOR = "distance_command"
    OBJECT_BODY = "cube_pair"

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, control_dt: float = 0.02):
        self.model = model
        self.data = data
        self.control_dt = control_dt
        self.body_id = self._name_id(mujoco.mjtObj.mjOBJ_BODY, self.OBJECT_BODY)
        self.object_joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, "object_pose")
        self.object_qpos_id = int(model.jnt_qposadr[self.object_joint_id])
        self.joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, self.DISTANCE_JOINT)
        self.actuator_id = self._name_id(mujoco.mjtObj.mjOBJ_ACTUATOR, self.DISTANCE_ACTUATOR)
        self.qpos_id = int(model.jnt_qposadr[self.joint_id])
        self.distance_range = model.jnt_range[self.joint_id].copy()
        self.target_distance = float(data.qpos[self.qpos_id])
        self.target_position = data.qpos[self.object_qpos_id:self.object_qpos_id + 3].copy()
        self.target_quat = data.qpos[self.object_qpos_id + 3:self.object_qpos_id + 7].copy()

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
        if action.size < 6:
            raise ValueError("Pincer action must contain at least 6 pose values")
        self.target_position += np.clip(action[:3], -0.05, 0.05)
        self.target_quat = self._quat_mul(self.target_quat, self._rotvec_to_quat(np.clip(action[3:6], -0.2, 0.2)))
        self.data.qpos[self.object_qpos_id:self.object_qpos_id + 3] = self.target_position
        self.data.qpos[self.object_qpos_id + 3:self.object_qpos_id + 7] = self.target_quat
        if action.size >= 7:
            self.target_distance = float(np.clip(self.target_distance + np.clip(action[6], -0.02, 0.02), *self.distance_range))
            self.data.ctrl[self.actuator_id] = self.target_distance
        mujoco.mj_forward(self.model, self.data)

    
    
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
