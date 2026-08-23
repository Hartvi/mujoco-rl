#!/usr/bin/env bash
/home/hartvi/venvs/ros2-torch/bin/python train_pincer_sb3.py \
    --total-timesteps 1000000 \
    --n-envs 16 \
    --episode-max-steps 500 \
    --output-dir runs/pincer_sb3 \
    --gamma 0.99 \
    --n-steps 512 \
    --env-type BowlingPickUp \
    --num-pins 1 \
    --pins-fallen 1
