from __future__ import annotations

import unittest
import warnings

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env as gymnasium_check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env

import pincer_controller
from bowling_scene import make_bowling_xml
from bowling_simple import BowlingSimple


class BowlingEnvironmentTest(unittest.TestCase):
    def test_bowling_scene_object_queries(self) -> None:
        env = BowlingSimple()
        ids = (
            env._pin_ids
            + env._cube_geom_ids
            + sum(env._pin_component_ids.values(), start=[])
        )

        for id in ids:
            assert id >= 0, f"Id: {id}"
            assert env.data.geom_xmat[id] is not None
            assert env.data.geom_xpos[id] is not None

    def test_gymnasium_and_sb3_environment_checks(self) -> None:
        env = BowlingSimple()
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
        env = BowlingSimple(max_steps=5)
        try:
            expected_qpos0 = env.bowling_scene.qpos0.copy()
            env.data.qpos[:] += 0.1
            env.data.qvel[:] = 1.0
            env.data.time = 7.0
            env._step_count = 4
            env._previous_fallen = 3
            env._rewarded_fallen_pins.add(env._pin_ids[0])
            env._last_action_dt = 1.0

            observation, info = env.reset(seed=123)

            expected_qpos0[env._ee.object_qpos_id : env._ee.object_qpos_id + 2] = (
                env.data.qpos[env._ee.object_qpos_id : env._ee.object_qpos_id + 2]
            )
            np.testing.assert_allclose(env.data.qpos, expected_qpos0)
            np.testing.assert_allclose(env.data.qvel, 0.0)
            self.assertEqual(env.data.time, 0.0)
            self.assertEqual(env._step_count, 0)
            self.assertEqual(env._previous_fallen, 0)
            self.assertEqual(env._rewarded_fallen_pins, set())
            self.assertEqual(env._last_action_dt, 0.0)
            self.assertEqual(info["fallen_pins"], 0)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertEqual(env._fallen_pins(), 0)
        finally:
            env.close()

    def test_seeded_reset_randomizes_pincer_near_pin_rack(self) -> None:
        env = BowlingSimple()
        try:
            env.reset(seed=123)
            first_xy = env.data.xpos[env._ee.body_id, :2].copy()
            pin_xy = np.asarray(
                [env.data.xpos[body_id, :2] for body_id in env._pin_ids]
            )
            rack_center = pin_xy.mean(axis=0)
            self.assertLessEqual(
                np.linalg.norm(first_xy - rack_center),
                env.START_RADIUS + 1e-9,
            )
            self.assertGreaterEqual(
                np.min(np.linalg.norm(pin_xy - first_xy, axis=1)),
                env.START_PIN_CLEARANCE - 1e-9,
            )

            env.reset(seed=123)
            np.testing.assert_allclose(env.data.xpos[env._ee.body_id, :2], first_xy)
            env.reset(seed=124)
            self.assertFalse(np.allclose(env.data.xpos[env._ee.body_id, :2], first_xy))
        finally:
            env.close()

    def test_observation_contains_fixed_order_pin_state(self) -> None:
        env = BowlingSimple()
        try:
            observation, _ = env.reset()
            state = observation["observation.state"]
            self.assertEqual(state.shape, (148,))
            pincer_position = env.data.xpos[env._ee.body_id]
            world_to_pincer = env.data.xmat[env._ee.body_id].reshape(3, 3).T
            for index, body_id in enumerate(env._pin_ids):
                start = env.PINCER_STATE_SIZE + index * env.PIN_STATE_SIZE
                pin_state = state[start : start + env.PIN_STATE_SIZE]
                expected_position = world_to_pincer @ (
                    env.data.xpos[body_id] - pincer_position
                )
                np.testing.assert_allclose(pin_state[:3], expected_position, atol=1e-7)
                np.testing.assert_allclose(
                    pin_state[3:7], [1.0, 0.0, 0.0, 0.0], atol=1e-7
                )
                self.assertEqual(pin_state[7], 0.0)
                np.testing.assert_allclose(pin_state[8:14], 0.0, atol=1e-7)
        finally:
            env.close()

    def test_pin_velocities_are_relative_to_pincer(self) -> None:
        env = BowlingSimple()
        try:
            env.reset()
            pincer_dof = int(env.bowling_scene.jnt_dofadr[env._ee.object_joint_id])
            pin_joint = mujoco.mj_name2id(
                env.bowling_scene, mujoco.mjtObj.mjOBJ_JOINT, "pin_1_free"
            )
            pin_dof = int(env.bowling_scene.jnt_dofadr[pin_joint])
            env.data.qvel[pincer_dof : pincer_dof + 6] = [
                0.25,
                0.5,
                0.75,
                1.0,
                1.0,
                1.0,
            ]
            env.data.qvel[pin_dof : pin_dof + 6] = [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ]
            mujoco.mj_forward(env.bowling_scene, env.data)
            state = env._get_observation()["observation.state"]
            start = env.PINCER_STATE_SIZE
            pincer_velocity = env._body_velocity(env._ee.body_id)
            pin_velocity = env._body_velocity(env._pin_ids[0])
            relative_position = (
                env.data.xpos[env._pin_ids[0]] - env.data.xpos[env._ee.body_id]
            )
            expected_linear = (
                pin_velocity[3:]
                - pincer_velocity[3:]
                - np.cross(pincer_velocity[:3], relative_position)
            )
            expected_angular = pin_velocity[:3] - pincer_velocity[:3]
            np.testing.assert_allclose(
                state[start + 8 : start + 11], expected_linear, atol=1e-7
            )
            np.testing.assert_allclose(
                state[start + 11 : start + 14], expected_angular, atol=1e-7
            )
        finally:
            env.close()

    def test_pin_fallen_flag_updates_in_its_fixed_block(self) -> None:
        env = BowlingSimple()
        try:
            env.reset()
            pin_index = 2
            joint_id = mujoco.mj_name2id(
                env.bowling_scene,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"pin_{pin_index + 1}_free",
            )
            qpos_adr = int(env.bowling_scene.jnt_qposadr[joint_id])
            env.data.qpos[qpos_adr + 3 : qpos_adr + 7] = [
                np.sqrt(0.5),
                np.sqrt(0.5),
                0.0,
                0.0,
            ]
            mujoco.mj_forward(env.bowling_scene, env.data)
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
        env = BowlingSimple(max_steps=2)
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

    def test_rotation_has_no_extra_penalty_beyond_action_penalty(self) -> None:
        env = BowlingSimple()
        try:
            env.reset()
            positive_action = np.zeros(7, dtype=np.float32)
            positive_action[3:6] = env.action_space.high[3:6]
            _, positive_reward, _, _, positive_info = env.step(positive_action)

            env.reset()
            negative_action = -positive_action
            _, negative_reward, _, _, negative_info = env.step(negative_action)

            self.assertIsInstance(positive_reward, float)
            self.assertIsInstance(positive_info["reward.rotation"], float)
            self.assertEqual(positive_info["reward.rotation"], 0.0)
            self.assertEqual(negative_info["reward.rotation"], 0.0)
            self.assertTrue(np.isscalar(negative_reward))
        finally:
            env.close()

    def test_distance_ignores_fallen_pins(self) -> None:
        env = BowlingSimple(num_pins=2)
        try:
            env.reset()
            fallen_pin_id, standing_pin_id = env._pin_ids
            fallen_joint_id = mujoco.mj_name2id(
                env.bowling_scene, mujoco.mjtObj.mjOBJ_JOINT, "pin_1_free"
            )
            fallen_qpos = int(env.bowling_scene.jnt_qposadr[fallen_joint_id])
            env.data.qpos[fallen_qpos + 3 : fallen_qpos + 7] = [
                np.sqrt(0.5),
                np.sqrt(0.5),
                0.0,
                0.0,
            ]
            env.data.qpos[env._ee.object_qpos_id : env._ee.object_qpos_id + 3] = (
                env.data.xpos[fallen_pin_id]
            )
            mujoco.mj_forward(env.bowling_scene, env.data)

            env._target_pin_id = standing_pin_id
            expected = np.linalg.norm(
                env.data.xpos[standing_pin_id]
                + np.array([0.0, 0.0, env.STRIKE_POINT_HEIGHT])
                - env.data.xpos[env._ee.body_id]
            )
            self.assertAlmostEqual(float(env._relevant_pin_distance()), float(expected))
        finally:
            env.close()

    def test_distance_targets_mid_pin_and_target_stays_stable(self) -> None:
        env = BowlingSimple(num_pins=2)
        try:
            env.reset(seed=1)
            target_pin_id = env._target_pin_id
            self.assertIsNotNone(target_pin_id)
            expected = np.linalg.norm(
                env.data.xpos[target_pin_id]
                + np.array([0.0, 0.0, env.STRIKE_POINT_HEIGHT])
                - env.data.xpos[env._ee.body_id]
            )
            self.assertAlmostEqual(float(env._relevant_pin_distance()), float(expected))

            env.data.qpos[env._ee.object_qpos_id] += 0.5
            mujoco.mj_forward(env.bowling_scene, env.data)
            self.assertEqual(env._target_pin_id, target_pin_id)
        finally:
            env.close()

    def test_distance_reward_tracks_signed_progress(self) -> None:
        env = BowlingSimple(max_steps=5)
        action = np.zeros(7, dtype=np.float32)
        try:
            env.reset()
            env._previous_pin_distance = 1.0
            env._relevant_pin_distance = lambda: 0.75  # type: ignore[method-assign]
            _, _, _, _, info = env.step(action)
            self.assertAlmostEqual(info["reward.distance"], 0.25 * env.DISTANCE_SCALE)

            _, _, _, _, info = env.step(action)
            self.assertEqual(info["reward.distance"], 0.0)

            env._relevant_pin_distance = lambda: 1.0  # type: ignore[method-assign]
            _, _, _, _, info = env.step(action)
            self.assertAlmostEqual(
                info["reward.distance"],
                -0.25 * env.DISTANCE_SCALE * env.AWAY_DISTANCE_MULTIPLIER,
            )
        finally:
            env.close()

    def test_action_penalty_uses_normalized_full_action(self) -> None:
        env = BowlingSimple(max_steps=5)
        try:
            env.reset()
            env._relevant_pin_distance = (  # type: ignore[method-assign]
                lambda: env._previous_pin_distance
            )
            action = env.action_space.high.copy()
            _, _, _, _, info = env.step(action)
            self.assertAlmostEqual(
                info["reward.action"],
                -env.ACTION_PENALTY_SCALE * action.size,
            )
        finally:
            env.close()

    def test_action_space_is_symmetric_and_normalized(self) -> None:
        env = BowlingSimple()
        try:
            np.testing.assert_array_equal(env.action_space.low, -1.0)
            np.testing.assert_array_equal(env.action_space.high, 1.0)
        finally:
            env.close()

    def test_jaw_has_no_dedicated_shaping_reward(self) -> None:
        env = BowlingSimple(max_steps=5)
        try:
            env.reset(seed=1)
            env._relevant_pin_distance = (  # type: ignore[method-assign]
                lambda: env._previous_pin_distance
            )
            closing = np.zeros(7, dtype=np.float32)
            closing[6] = -1.0
            _, _, _, _, close_info = env.step(closing)

            env.reset(seed=1)
            env._relevant_pin_distance = (  # type: ignore[method-assign]
                lambda: env._previous_pin_distance
            )
            opening = -closing
            _, _, _, _, open_info = env.step(opening)

            self.assertEqual(close_info["reward.open_close"], 0.0)
            self.assertEqual(open_info["reward.open_close"], 0.0)
        finally:
            env.close()

    def test_newly_fallen_pin_uses_large_bonus(self) -> None:
        env = BowlingSimple(max_steps=5)
        try:
            env.reset()
            env._fallen_pin_ids = lambda: {  # type: ignore[method-assign]
                env._pin_ids[0]
            }
            _, _, _, _, info = env.step(np.zeros(7, dtype=np.float32))
            self.assertEqual(info["reward.fallen_pins"], env.NEWLY_FALLEN_REWARD)
        finally:
            env.close()

    def test_each_pin_fall_reward_is_paid_only_once(self) -> None:
        env = BowlingSimple(max_steps=5)
        action = np.zeros(7, dtype=np.float32)
        try:
            env.reset()
            fallen_ids: set[int] = {env._pin_ids[0]}
            env._fallen_pin_ids = lambda: fallen_ids  # type: ignore[method-assign]

            _, _, _, _, first_info = env.step(action)
            fallen_ids = set()
            env.step(action)
            fallen_ids = {env._pin_ids[0]}
            _, _, _, _, second_info = env.step(action)

            self.assertEqual(first_info["reward.fallen_pins"], env.NEWLY_FALLEN_REWARD)
            self.assertEqual(second_info["reward.fallen_pins"], 0.0)
        finally:
            env.close()

    def test_pin_touch_reward_is_paid_once_per_pin_per_episode(self) -> None:
        env = BowlingSimple(max_steps=5)
        action = np.zeros(7, dtype=np.float32)
        try:
            env.reset()
            pin_id = env._pin_ids[0]
            env._touching_pins = lambda: {pin_id}  # type: ignore[method-assign]

            _, _, _, _, first_info = env.step(action)
            _, _, _, _, second_info = env.step(action)

            self.assertEqual(first_info["newly_touched_pins"], 1)
            self.assertEqual(first_info["reward.pin_touch"], env.PIN_TOUCH_REWARD)
            self.assertEqual(second_info["newly_touched_pins"], 0)
            self.assertEqual(second_info["reward.pin_touch"], 0.0)
        finally:
            env.close()

    def test_success_is_termination_not_truncation(self) -> None:
        env = BowlingSimple(max_steps=5)
        action = np.zeros(7, dtype=np.float32)
        try:
            env.reset()
            env._fallen_pin_ids = lambda: set(  # type: ignore[method-assign]
                env._pin_ids
            )
            _, _, terminated, truncated, info = env.step(action)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["success"])
            self.assertEqual(info["reward.success"], env.SUCCESS_REWARD)
        finally:
            env.close()

    def test_motion_commands_are_rate_limited(self) -> None:
        env = BowlingSimple()
        try:
            observation, _ = env.reset()
            initial_position = observation["observation.state"][:3].copy()
            initial_target_position = env._ee.target_position.copy()
            initial_target_distance = env._ee.target_distance

            observation, _, _, _, _ = env.step(env.action_space.high.copy())
            state = observation["observation.state"]

            np.testing.assert_allclose(
                env._ee.target_position - initial_target_position,
                env._ee.MAX_LINEAR_SPEED * env.control_dt,
                atol=1e-7,
            )
            self.assertAlmostEqual(
                env._ee.target_distance - initial_target_distance,
                env._ee.MAX_DISTANCE_SPEED * env.control_dt,
                places=7,
            )
            self.assertTrue(np.all(state[:3] > initial_position))
            self.assertAlmostEqual(float(state[7]), env.control_dt, places=7)
        finally:
            env.close()

    @staticmethod
    def _contact_names(env: BowlingSimple) -> set[frozenset[str | None]]:
        return {
            frozenset(
                (
                    mujoco.mj_id2name(
                        env.bowling_scene,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        int(contact.geom1),
                    ),
                    mujoco.mj_id2name(
                        env.bowling_scene,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        int(contact.geom2),
                    ),
                )
            )
            for contact in env.data.contact
        }

    def test_ground_contact_blocks_downward_commands_and_changes_reward(
        self,
    ) -> None:
        env = BowlingSimple(max_steps=200)
        try:
            env.reset()
            zero_action = np.zeros(7, dtype=np.float32)
            _, _, _, _, info = env.step(zero_action)
            expected_ground_reward = -env.GROUND_PENALTY_SCALE * max(
                0.0,
                env.MIN_GROUND_CLEARANCE - info["distance.ground_clearance"],
            )
            self.assertAlmostEqual(
                info["reward.ground_clearance"], expected_ground_reward
            )

            downward_action = zero_action.copy()
            downward_action[2] = env.action_space.low[2]
            for _ in range(100):
                _, _, _, _, info = env.step(downward_action)

            self.assertGreater(info["distance.ground_clearance"], -0.005)
            self.assertLess(info["reward.ground_clearance"], 0.0)
            self.assertGreater(env.data.xpos[env._ee.body_id, 2], 0.0)
        finally:
            env.close()

    def test_pincer_contact_pushes_pin_instead_of_phasing_through(self) -> None:
        env = BowlingSimple(num_pins=1, max_steps=200)
        try:
            env.reset()
            pin_id = env._pin_ids[0]
            initial_pin_x = float(env.data.xpos[pin_id, 0])
            env.data.qpos[env._ee.object_qpos_id : env._ee.object_qpos_id + 3] = [
                initial_pin_x - 0.3,
                env.data.xpos[pin_id, 1],
                0.05,
            ]
            env.data.qvel[env._ee.object_dof_id : env._ee.object_dof_id + 6] = 0.0
            env._ee.sync_target_to_pose()
            forward_action = np.zeros(7, dtype=np.float32)
            forward_action[0] = env.action_space.high[0]
            max_acceleration = 0.0

            for _ in range(130):
                env.step(forward_action)
                self.assertTrue(np.all(np.isfinite(env.data.qacc)))
                max_acceleration = max(
                    max_acceleration, float(np.max(np.abs(env.data.qacc)))
                )

            self.assertGreater(
                float(env.data.xpos[pin_id, 0]) - initial_pin_x,
                0.05,
            )
            self.assertLess(max_acceleration, 1e5)
            self.assertTrue(
                np.all(
                    env.bowling_scene.dof_damping[
                        env._ee.object_dof_id : env._ee.object_dof_id + 6
                    ]
                    > 0.0
                )
            )
        finally:
            env.close()

    def test_internal_collision_excluded_and_external_contacts_work(
        self,
    ) -> None:
        env = BowlingSimple()
        try:
            env.reset()
            self.assertNotIn(frozenset(("cube_1", "cube_2")), self._contact_names(env))

            env.data.qpos[env._ee.object_qpos_id + 2] = 0.005
            mujoco.mj_forward(env.bowling_scene, env.data)
            ground_contacts = self._contact_names(env)
            self.assertTrue(
                all(
                    frozenset(("ground", cube)) in ground_contacts
                    for cube in ("cube_1", "cube_2")
                )
            )

            env.reset()
            pin_position = env.data.xpos[env._pin_ids[0]].copy()
            env.data.qpos[env._ee.object_qpos_id : env._ee.object_qpos_id + 3] = (
                pin_position + np.array([0.0, 0.0, 0.08])
            )
            mujoco.mj_forward(env.bowling_scene, env.data)
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
