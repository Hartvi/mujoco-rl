# Pincer Bowling RL TODO

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
- [ ] Normalize observations with SB3 `VecNormalize` during the SB3 rewrite.
- [x] Retain delta time for future variable-rate control.

## 3. Redesign and test the reward

- [ ] Target the nearest upright pin, not an already-fallen pin.
- [ ] Keep a dense approach term, such as negative distance to the target pin.
- [ ] Reward only newly fallen pins rather than repeatedly rewarding every fallen pin.
- [ ] Increase the newly-fallen bonus enough to dominate incidental per-step distance costs.
- [ ] Clarify the open/close shaping objective:
  - penalize unnecessary jaw motion;
  - optionally reward closing near an upright pin;
  - avoid discouraging useful grasping or contact.
- [ ] Consider a small control or movement penalty.
- [ ] Consider a success bonus when all requested pins have fallen.
- [ ] Log every reward component separately.
- [ ] Plot reward components and pin success versus training steps.
- [ ] Test the reward manually with scripted trajectories before training.

## 4. Normalize the action interface

- [ ] Expose a symmetric `[-1, 1]` action space to the policy.
- [ ] Rescale normalized translation commands internally to the `0.05 m/s` limit.
- [ ] Rescale normalized rotation commands internally to the selected angular-speed limit.
- [ ] Rescale normalized jaw commands internally to the `0.05 m/s` opening-speed limit.
- [ ] Keep action semantics documented:
  `[vx, vy, vz, wx, wy, wz, jaw_velocity]`.
- [ ] Verify limits remain correct if the control timestep changes.

## 5. Replace handwritten PPO with Stable-Baselines3

- [ ] Preserve the current raw trainer temporarily as a reference.
- [x] Add a new SB3 training entry point.
- [ ] Use `MultiInputPolicy` while observations remain a `Dict`; otherwise use `MlpPolicy`.
- [ ] Match the desired actor and critic architecture with `policy_kwargs`.
- [ ] Use `Monitor` for episode metrics.
- [ ] Add TensorBoard logging.
- [ ] Add checkpoint and best-model callbacks.
- [ ] Add a separate deterministic evaluation environment and `EvalCallback`.
- [ ] Save all configuration and normalization statistics with each model.
- [ ] Add resume-training and inference/evaluation commands.

## 6. Parallelize rollout collection

- [ ] Start with 4-8 headless environments using `SubprocVecEnv`.
- [ ] Keep one separate single environment for human rendering and debugging.
- [ ] Remember that SB3 rollout size is `n_steps * n_envs`.
- [ ] Select `n_steps` and `batch_size` so the total rollout divides cleanly into minibatches.
- [ ] Benchmark `DummyVecEnv` against `SubprocVecEnv`; use whichever is faster for this MuJoCo environment.
- [ ] Compare CPU and CUDA policy updates; a small MLP may be faster on CPU.
- [ ] Do not expect high VRAM use from a small MLP.

## 7. Handle termination and truncation correctly

- [ ] Give the environment a dedicated `episode_max_steps` setting.
- [ ] Let SB3 choose rollout length independently through `n_steps`.
- [ ] Preserve Gymnasium's separate `terminated` and `truncated` signals.
- [ ] Confirm value bootstrapping occurs on time-limit truncation.
- [ ] Add tests for success termination, timeout truncation, and reset after each case.

## 8. Evaluation and reproducibility

- [ ] Seed Python, NumPy, Torch, and every environment worker.
- [ ] Randomize initial pincer pose and/or pin conditions within controlled bounds.
- [ ] Track success rate, fallen pins per episode, episode return, episode length, and target distance.
- [ ] Compare against random, scripted, and no-op baselines.
- [ ] Evaluate across multiple seeds.
- [ ] Record periodic evaluation videos rather than rendering every training environment.
- [ ] Add a short smoke-training test suitable for regular regression checks.

## Suggested implementation order

1. Fix observation and target-pin selection.
2. Normalize actions and verify physical speed limits.
3. Tune and unit-test reward components.
4. Separate episode timeout from rollout length.
5. Add the SB3 single-environment trainer.
6. Add evaluation, logging, and checkpoints.
7. Add vectorized environments.
8. Tune PPO and reward scales using multiple seeds.
