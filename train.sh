#!/usr/bin/env bash
/home/hartvi/venvs/ros2-torch/bin/python train_pincer_sb3.py \
    --total-timesteps 1000000 \
    --output-dir runs/pincer_sb3 \
    --progress-bar
