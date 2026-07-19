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
