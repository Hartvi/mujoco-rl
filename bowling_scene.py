"""MuJoCo scene construction for the bowling task."""
from __future__ import annotations


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


def _panda_xml() -> str:
    """A self-contained Panda-shaped 7-DOF arm.

    The geometry is intentionally primitive so the environment has no mesh
    download/runtime dependency. Joint and site names match a Panda-style API.
    """
    colors = [
        "0.15 0.15 0.18 1", "0.85 0.85 0.88 1", "0.15 0.15 0.18 1",
        "0.85 0.85 0.88 1", "0.15 0.15 0.18 1", "0.85 0.85 0.88 1",
        "0.15 0.15 0.18 1",
    ]
    axes = ["0 0 1", "0 1 0", "0 1 0", "0 0 1", "0 1 0", "0 0 1", "0 1 0"]
    # A compact serial arm along +x. The IK controller operates on its site.
    tail = '''
          <site name="panda_hand" pos="0.12 0 0" size="0.035" rgba="0.1 0.8 0.1 1"/>
          <geom name="panda_gripper_palm" type="box" pos="0.08 0 0" size="0.08 0.10 0.045" rgba="0.12 0.12 0.14 1"/>
          <body name="panda_finger_left" pos="0.16 0.07 0">
            <joint name="panda_finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
            <geom type="box" pos="0.05 0 0" size="0.06 0.018 0.018" rgba="0.75 0.75 0.78 1"/>
          </body>
          <body name="panda_finger_right" pos="0.16 -0.07 0">
            <joint name="panda_finger_joint2" type="slide" axis="0 -1 0" range="0 0.04"/>
            <geom type="box" pos="0.05 0 0" size="0.06 0.018 0.018" rgba="0.75 0.75 0.78 1"/>
          </body>
        </body>'''
    # Construct the nested serial chain explicitly.
    chain = ""
    for i, (axis, color) in enumerate(zip(axes, colors), start=1):
        chain += f'''
        <body name="panda_link{i}" pos="0.19 0 0">
          <joint name="panda_joint{i}" type="hinge" axis="{axis}" range="-2.9 2.9" damping="2"/>
          <geom name="panda_link{i}_geom" type="capsule" fromto="0 0 0 0.19 0 0" size="0.055" rgba="{color}" mass="0.35"/>'''
    chain += tail
    # The hand fragment closes link 7; close the remaining six links here.
    chain += "</body>" * 6
    return f'''
    <body name="panda" pos="-3.15 0 0.35">
      <geom name="panda_base" type="cylinder" size="0.18 0.12" rgba="0.12 0.12 0.14 1"/>
      {chain}
    </body>'''


def _pincer_xml() -> str:
    return """
    <body name="cube_pair" pos="0.3 0 0.05" gravcomp="1">
      <freejoint name="object_pose"/>
      <geom name="pincer_center_mass" type="box" size="0.01 0.01 0.01" mass="0.008" contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <body name="cube_1_body" pos="-0.01 0 0">
        <geom name="cube_1" type="box" size="0.01 0.01 0.01" mass="0.000001" condim="3" rgba="0.2 0.5 0.9 1"/>
      </body>
      <body name="cube_2_body" pos="0.01 0 0">
        <joint name="cube_distance" type="slide" axis="1 0 0" limited="true" range="0.02 0.12" ref="0.02" damping="0.2"/>
        <geom name="cube_2" type="box" size="0.01 0.01 0.01" mass="0.000001" condim="3" rgba="0.9 0.3 0.2 1"/>
      </body>
    </body>
    """


def make_bowling_xml(include_pincer: bool = False) -> str:
    positions = []
    spacing = 0.27
    for row in range(4):
        x = 1.25 + row * spacing * 0.86
        for column in range(row + 1):
            positions.append((x, (column - row / 2.0) * spacing))
    pins = "\n".join(_pin_xml(i + 1, x, y) for i, (x, y) in enumerate(positions))
    return f'''<mujoco model="bowling_panda">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81" iterations="80" solver="Newton"/>
  <size njmax="2000" nconmax="1000"/>
  <visual><headlight diffuse="0.8 0.8 0.8" ambient="0.25 0.25 0.25"/><map znear="0.01" zfar="30"/></visual>
  <asset>
    <texture name="floor_texture" type="2d" builtin="checker" width="512" height="512" rgb1="0.72 0.55 0.34" rgb2="0.58 0.40 0.23"/>
    <material name="lane" texture="floor_texture" texrepeat="6 2"/>
  </asset>
  <worldbody>
    <light name="overhead" pos="0 0 6" dir="0 0 -1" directional="true"/>
    <camera name="laptop_camera" pos="-4 -3 2.8" mode="targetbody" target="ball" fovy="55"/>
    <camera name="phone_camera" pos="-1.2 2.2 1.3" mode="targetbody" target="ball" fovy="60"/>
    <geom name="ground" type="plane" size="12 6 0.1" material="lane" friction="0.7 0.01 0.005"/>
    <geom name="back_wall" type="box" size="0.12 2.5 0.5" pos="3 0 0.5" rgba="0.25 0.25 0.28 1"/>
    <body name="ball" pos="-2 0 0.14">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.14" mass="7" rgba="0.03 0.08 0.65 1" friction="0.5 0.02 0.01"/>
    </body>
    {pins}
    {_pincer_xml() if include_pincer else _panda_xml()}
  </worldbody>
  <contact>
    <exclude body1="cube_pair" body2="cube_2_body"/>
  </contact>

  <actuator>
    {((''.join(f'<position name="panda_motor{i}" joint="panda_joint{i}" kp="150" kv="20" ctrlrange="-2.9 2.9"/>' for i in range(1, 8)) + '<position name="panda_finger_motor1" joint="panda_finger_joint1" kp="100" ctrlrange="0 0.04"/><position name="panda_finger_motor2" joint="panda_finger_joint2" kp="100" ctrlrange="0 0.04"/>') if not include_pincer else '<position name="distance_command" joint="cube_distance" kp="100" ctrllimited="true" ctrlrange="0.02 0.12" forcelimited="true" forcerange="-20 20"/>')}
  </actuator>
</mujoco>'''
