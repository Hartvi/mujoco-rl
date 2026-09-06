#!/usr/bin/env bash
/home/hartvi/venvs/ros2-torch/bin/python evaluate_pincer_sb3.py \
      --episodes 1 \
      --episode-max-steps 500 \
      --env-type BowlingPickUp \
      --render \
      --real-time \
      --seed "$RANDOM" \
      --model runs/pincer_pickup_sb3/best/best_model.zip \
      --vecnormalize runs/pincer_pickup_sb3/best/vecnormalize.pkl \
      --num-pins 1
