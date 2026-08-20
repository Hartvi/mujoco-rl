# Pincer Bowling RL TODO

### 0. Developer
- [ ] fix pre-commit & enforce proper formatting
- [ ] add auto-generate stubs script
- [ ] add requirements.txt

## 1. Validate the environment

- [x] Run both Gymnasium and Stable-Baselines3 `check_env` checks.
- [x] Keep `bowling_scene.py` as the single source of truth for the pincer XML.
- [x] Verify reset restores the intended pincer pose, cube opening, pins, velocities, simulation time, and reward state.
- [x] Keep episode duration independent from PPO rollout length.
- [x] Confirm `terminated` is used only for task success and `truncated` only for the episode time limit.
- [x] Test pincer translation, rotation, and opening limits against simulated time.
- [x] Test internal collision exclusions and contacts with pins and the ground.

## 2. Improve the observation

- [x] Add enough pin state to make the task observable:
  - pin positions relative to the pincer;
  - pin orientation;
  - fallen/upright flags;
  - pin linear and angular velocities relative to the pincer.
- [x] Use all-pin observations in fixed pin_1 through pin_10 order.
- [x] Keep pincer position, orientation, opening, and action delta time.
- [x] Use fixed ordering and fixed dimensions for every pin.
- [x] Normalize observations with SB3 `VecNormalize` during the SB3 rewrite.
- [x] Retain delta time for future variable-rate control.

## 3. Redesign and test the reward

- [x] Target the nearest upright pin, not an already-fallen pin.
- [x] Keep a dense approach term, such as negative distance to the target pin.
- [x] Reward only newly fallen pins rather than repeatedly rewarding every fallen pin.
- [x] Increase the newly-fallen bonus enough to dominate incidental per-step distance costs.
- [x] Clarify the open/close shaping objective:
  - penalize unnecessary jaw motion;
  - optionally reward closing near an upright pin;
  - avoid discouraging useful grasping or contact.
- [x] Add a small normalized action penalty.
- [x] Add a success bonus when all requested pins have fallen.
- [x] Log every reward component separately.
- [x] Plot reward components and pin success versus training steps in TensorBoard.
- [x] Add a scripted nearest-pin trajectory for reward and policy comparisons.

## 4. Normalize the action interface

- [x] Expose a symmetric `[-1, 1]` action space to the policy.
- [x] Rescale normalized translation commands internally to the current `0.5 m/s` limit.
- [x] Rescale normalized rotation commands internally to the `0.5 rad/s` limit.
- [x] Rescale normalized jaw commands internally to the `0.05 m/s` opening-speed limit.
- [x] Keep action semantics documented:
  `[vx, vy, vz, wx, wy, wz, jaw_velocity]`.
- [x] Verify limits remain correct if the control timestep changes.

## 5. Replace handwritten PPO with Stable-Baselines3

- [x] Preserve the current raw trainer temporarily as a reference.
- [x] Add a new SB3 training entry point.
- [x] Flatten observations and use `MlpPolicy`.
- [x] Match the desired actor and critic architecture with `policy_kwargs`.
- [x] Use `Monitor` for episode metrics.
- [x] Add TensorBoard logging.
- [x] Add checkpoint and best-model callbacks.
- [x] Add a separate deterministic evaluation environment and `EvalCallback`.
- [x] Save all configuration and normalization statistics with each model.
- [x] Add resume-training and inference/evaluation commands.

## 6. Parallelize rollout collection

- [x] Start with 4 headless environments using `SubprocVecEnv`.
- [x] Keep one separate single environment for human rendering and debugging.
- [x] Remember that SB3 rollout size is `n_steps * n_envs`.
- [x] Select `n_steps` and `batch_size` so the total rollout divides cleanly into minibatches.
- [ ] Benchmark `DummyVecEnv` against `SubprocVecEnv`; use whichever is faster for this MuJoCo environment.
- [ ] Compare CPU and CUDA policy updates; a small MLP may be faster on CPU.
- [ ] Do not expect high VRAM use from a small MLP.

## 7. Handle termination and truncation correctly

- [ ] Give the environment a dedicated `episode_max_steps` setting.
- [x] Let SB3 choose rollout length independently through `n_steps`.
- [x] Preserve Gymnasium's separate `terminated` and `truncated` signals.
- [ ] Confirm value bootstrapping occurs on time-limit truncation.
- [x] Add tests for success termination, timeout truncation, and reset after each case.

## 8. Evaluation and reproducibility

- [x] Seed Python, NumPy, Torch, and every environment worker.
- [x] Randomize initial pincer pose and/or pin conditions within controlled bounds.
- [x] Track success rate, fallen pins per episode, episode return, episode length, and target distance.
- [x] Compare against random, scripted, and no-op baselines.
- [ ] Evaluate across multiple seeds.
- [ ] Record periodic evaluation videos rather than rendering every training environment.
- [x] Add a short smoke-training test suitable for regular regression checks.

## Suggested implementation order

1. Fix observation and target-pin selection.
2. Normalize actions and verify physical speed limits.
3. Tune and unit-test reward components.
4. Separate episode timeout from rollout length.
5. Add the SB3 single-environment trainer.
6. Add evaluation, logging, and checkpoints.
7. Add vectorized environments.
8. Tune PPO and reward scales using multiple seeds.
