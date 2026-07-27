/home/hartvi/venvs/ros2-torch/bin/python evaluate_pincer_sb3.py \
      --episodes 1 \
      --episode-max-steps 1500 \
      --render \
      --real-time \
      --seed "$RANDOM" \
      --model $1 \
      --vecnormalize $2

