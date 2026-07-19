from __future__ import annotations

import unittest
import warnings

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env as gymnasium_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

import pincer_controller
from bowling_scene import make_bowling_xml
from bowling_simple import BowlingEnv


class BowlingEnvironmentTest(unittest.TestCase):
    def test_gymnasium_and_sb3_environment_checks(self) -> None:
        env = BowlingEnv()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gymnasium_check_env(env, skip_render_check=True)
                sb3_check_env(env, warn=True, skip_render_check=True)
        finally:
            env.close()

    def test_bowling_scene_is_the_only_pincer_xml_source(self) -> None:
        self.assertFalse(hasattr(pincer_controller, "_pincer_xml"))
        self.assertEqual(
            make_bowling_xml(include_pincer=True).count('name="cube_pair"'), 1
        )

    def test_reset_restores_full_episode_state(self) -> None:
        env = BowlingEnv(max_steps=5)
        try:
            expected_qpos0 = env.model.qpos0.copy()
            env.data.qpos[:] += 0.1
            env.data.qvel[:] = 1.0
            env.data.time = 7.0
            env._step_count = 4
            env._previous_fallen = 3
            env._last_action_dt = 1.0

            observation, info = env.reset(seed=123)

            np.testing.assert_allclose(env.data.qpos, expected_qpos0)
            np.testing.assert_allclose(env.data.qvel, 0.0)
            self.assertEqual(env.data.time, 0.0)
            self.assertEqual(env._step_count, 0)
            self.assertEqual(env._previous_fallen, 0)
            self.assertEqual(env._last_action_dt, 0.0)
            self.assertEqual(info["fallen_pins"], 0)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(env._fallen_pins(), 0)
        finally:
            env.close()

    def test_observation_contains_fixed_order_pin_state(self) -> None:
        env = BowlingEnv()
        try:
            observation, _ = env.reset()
            state = observation["observation.state"]
            self.assertEqual(state.shape, (148,))
            pincer_position = env.data.xpos[env._ee.body_id]
            world_to_pincer = env.data.xmat[env._ee.body_id].reshape(3, 3).T
            for index, body_id in enumerate(env._pin_ids):
                start = env.PINCER_STATE_SIZE + index * env.PIN_STATE_SIZE
                pin_state = state[start:start + env.PIN_STATE_SIZE]
                expected_position = world_to_pincer @ (env.data.xpos[body_id] - pincer_position)
                np.testing.assert_allclose(pin_state[:3], expected_position, atol=1e-7)
                np.testing.assert_allclose(pin_state[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-7)
                self.assertEqual(pin_state[7], 0.0)
                np.testing.assert_allclose(pin_state[8:14], 0.0, atol=1e-7)
        finally:
            env.close()

    def test_pin_velocities_are_relative_to_pincer(self) -> None:
        env = BowlingEnv()
        try:
            env.reset()
            pincer_dof = int(env.model.jnt_dofadr[env._ee.object_joint_id])
            pin_joint = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "pin_1_free")
            pin_dof = int(env.model.jnt_dofadr[pin_joint])
            env.data.qvel[pincer_dof:pincer_dof + 6] = [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]
            env.data.qvel[pin_dof:pin_dof + 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            mujoco.mj_forward(env.model, env.data)
            state = env._get_observation()["observation.state"]
            start = env.PINCER_STATE_SIZE
            pincer_velocity = env._body_velocity(env._ee.body_id)
            pin_velocity = env._body_velocity(env._pin_ids[0])
            relative_position = env.data.xpos[env._pin_ids[0]] - env.data.xpos[env._ee.body_id]
            expected_linear = pin_velocity[3:] - pincer_velocity[3:] - np.cross(pincer_velocity[:3], relative_position)
            expected_angular = pin_velocity[:3] - pincer_velocity[:3]
            np.testing.assert_allclose(state[start + 8:start + 11], expected_linear, atol=1e-7)
            np.testing.assert_allclose(state[start + 11:start + 14], expected_angular, atol=1e-7)
        finally:
            env.close()

    def test_pin_fallen_flag_updates_in_its_fixed_block(self) -> None:
        env = BowlingEnv()
        try:
            env.reset()
            pin_index = 2
            joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, f"pin_{pin_index + 1}_free")
            qpos_adr = int(env.model.jnt_qposadr[joint_id])
            env.data.qpos[qpos_adr + 3:qpos_adr + 7] = [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0]
            mujoco.mj_forward(env.model, env.data)
            state = env._get_observation()["observation.state"]
            start = env.PINCER_STATE_SIZE + pin_index * env.PIN_STATE_SIZE
            self.assertEqual(state[start + 7], 1.0)
            for other_index in range(env.num_pins):
                if other_index == pin_index:
                    continue
                other_start = env.PINCER_STATE_SIZE + other_index * env.PIN_STATE_SIZE
                self.assertEqual(state[other_start + 7], 0.0)
        finally:
            env.close()

    def test_timeout_is_truncation_not_termination(self) -> None:
        env = BowlingEnv(max_steps=2)
        action = np.zeros(7, dtype=np.float32)
        try:
            env.reset()
            _, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            _, _, terminated, truncated, _ = env.step(action)
            self.assertFalse(terminated)
            self.assertTrue(truncated)
        finally:
            env.close()

    def test_rotation_reward_is_a_direction_independent_scalar_penalty(self) -> None:
        env = BowlingEnv()
        try:
            env.reset()
            positive_action = np.zeros(7, dtype=np.float32)
            positive_action[3:6] = env.action_space.high[3:6]
            _, positive_reward, _, _, positive_info = env.step(positive_action)

            env.reset()
            negative_action = -positive_action
            _, negative_reward, _, _, negative_info = env.step(negative_action)

            expected = -float(np.linalg.norm(positive_action[3:6]))
            self.assertIsInstance(positive_reward, float)
            self.assertIsInstance(positive_info["reward.rotation"], float)
            self.assertAlmostEqual(positive_info["reward.rotation"], expected)
            self.assertAlmostEqual(negative_info["reward.rotation"], expected)
            self.assertTrue(np.isscalar(negative_reward))
        finally:
            env.close()

    def test_success_is_termination_not_truncation(self) -> None:
        env = BowlingEnv(max_steps=5)
        action = np.zeros(7, dtype=np.float32)
        try:
            env.reset()
            env._fallen_pins = lambda: env.num_pins
            _, _, terminated, truncated, info = env.step(action)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["success"])
        finally:
            env.close()

    def test_motion_limits_match_control_timestep(self) -> None:
        env = BowlingEnv()
        try:
            observation, _ = env.reset()
            initial_position = observation["observation.state"][:3].astype(np.float64)
            initial_distance = float(observation["observation.state"][6])
            observation, _, _, _, _ = env.step(env.action_space.high.copy())
            state = observation["observation.state"]

            np.testing.assert_allclose(
                state[:3] - initial_position,
                env._ee.MAX_LINEAR_SPEED * env.control_dt,
                atol=1e-7,
            )
            self.assertAlmostEqual(
                float(state[6]) - initial_distance,
                env._ee.MAX_DISTANCE_SPEED * env.control_dt,
                places=7,
            )
            self.assertAlmostEqual(float(state[7]), env.control_dt, places=7)
            self.assertAlmostEqual(
                np.linalg.norm(state[3:6]) / env.control_dt,
                np.sqrt(3.0) * env._ee.MAX_ANGULAR_SPEED,
                places=5,
            )
        finally:
            env.close()

    @staticmethod
    def _contact_names(env: BowlingEnv) -> set[frozenset[str | None]]:
        return {
            frozenset(
                (
                    mujoco.mj_id2name(
                        env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)
                    ),
                    mujoco.mj_id2name(
                        env.model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)
                    ),
                )
            )
            for contact in env.data.contact
        }

    def test_internal_collision_excluded_and_external_contacts_work(self) -> None:
        env = BowlingEnv()
        try:
            env.reset()
            self.assertNotIn(
                frozenset(("cube_1", "cube_2")), self._contact_names(env)
            )

            env._ee.target_position[2] = 0.005
            env._ee.hold_pose()
            ground_contacts = self._contact_names(env)
            self.assertTrue(
                all(
                    frozenset(("ground", cube)) in ground_contacts
                    for cube in ("cube_1", "cube_2")
                )
            )

            env.reset()
            pin_position = env.data.xpos[env._pin_ids[0]].copy()
            env._ee.target_position[:] = pin_position + np.array([0.0, 0.0, 0.08])
            env._ee.hold_pose()
            pin_contacts = self._contact_names(env)
            self.assertTrue(
                any(
                    "cube_" in (name or "") and "pin_1" in (other or "")
                    for pair in pin_contacts
                    for name in pair
                    for other in pair
                )
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
