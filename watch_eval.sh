/home/hartvi/venvs/ros2-torch/bin/python evaluate_pincer_sb3.py \
      --episodes 1 \
      --episode-max-steps 1500 \
      --render \
      --real-time \
      --seed "$RANDOM" \
      --model runs/pincer_sb3/best/best_model.zip \
      --vecnormalize  runs/pincer_sb3/best/vecnormalize.pkl

