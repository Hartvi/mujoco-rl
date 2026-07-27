# Pincer Bowling RL

The primary training path uses Stable-Baselines3 PPO. Policy actions are
normalized to `[-1, 1]` with this fixed layout:

`[vx, vy, vz, wx, wy, wz, jaw_velocity]`

The environment maps those commands to the current physical limits of 0.5 m/s
translation, 0.5 rad/s rotation, and 0.05 m/s jaw motion.

## Train

```bash
./train_headless.sh
```

Artifacts are written under `runs/pincer_sb3`: the final model, observation
normalization statistics, configuration, periodic checkpoints, best model,
evaluation data, monitor logs, and TensorBoard events.

Training uses four parallel MuJoCo workers by default. Override this with
`--n-envs`; use `--n-envs 1` for interactive debugging. PPO collects
`n_steps * n_envs` samples per rollout, and `batch_size` must divide that total.

```bash
tensorboard --logdir runs/pincer_sb3/tensorboard
```

## Evaluate

```bash
./eval_baselines.sh
```

This compares the deterministic checkpoint with scripted-nearest-pin, random,
and no-op baselines. To render one episode:

```bash
./watch_eval.sh
```

## Resume

```bash
python train_pincer_sb3.py \
  --resume runs/pincer_sb3/final_model.zip \
  --vecnormalize runs/pincer_sb3/vecnormalize.pkl \
  --output-dir runs/pincer_sb3-resumed
```

The older `train_pincer_mlp.py` and `evaluate_pincer_mlp.py` files remain only
as references for the former handwritten PPO implementation.
