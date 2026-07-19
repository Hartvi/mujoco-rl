from __future__ import annotations
import argparse
import csv
import torch
from bowling_simple import BowlingEnv
from pincer_mlp import PincerMLP

def train(args):
    env = BowlingEnv(render_mode="human" if args.render else None, max_steps=args.episode_max_steps)
    log_file = open(args.log_file, "w", newline="")
    log_writer = csv.DictWriter(log_file, fieldnames=["update", "mean_reward", "distance_reward", "fallen_reward", "open_close_reward", "mean_pin_distance", "max_fallen_pins"])
    log_writer.writeheader()
    device = torch.device(args.device)
    model = PincerMLP(observation_dim=env.observation_space["observation.state"].shape[0], action_low=env.action_space.low, action_high=env.action_space.high).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    observation, _ = env.reset(seed=args.seed)
    total_steps = 0

    for update in range(args.updates):
        states, raw_actions, old_log_probs, rewards, dones, values = [], [], [], [], [], []
        distance_rewards, fallen_rewards, open_close_rewards, pin_distances, fallen_counts = [], [], [], [], []

        for step in range(args.horizon):
            state = torch.as_tensor(model.flatten_observation(observation), dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value, raw = model.action_and_value(state)
            env_action = action[0].cpu().numpy()
            observation, reward, terminated, truncated, info = env.step(env_action)
            total_steps += 1
            if args.log_actions and (total_steps % args.log_every == 0):
                pincer_state = observation["observation.state"][:8]
                print(
                    f"[ACTION] step={total_steps} action={env_action.round(3)} "
                    f"pincer={pincer_state.round(3)} reward={reward:.3f} "
                    f"fallen={info['fallen_pins']}",
                    flush=True,
                )
            distance_rewards.append(info["reward.distance"])
            fallen_rewards.append(info["reward.fallen_pins"])
            open_close_rewards.append(info["reward.open_close"])
            pin_distances.append(info["distance.relevant_pin"])
            fallen_counts.append(info["fallen_pins"])
            states.append(state[0].cpu()); raw_actions.append(raw[0].cpu()); old_log_probs.append(log_prob[0].cpu())
            rewards.append(reward); dones.append(float(terminated or truncated)); values.append(value[0].cpu())
            if terminated or truncated:
                observation, _ = env.reset()

        states = torch.stack(states).to(device); raw_actions = torch.stack(raw_actions).to(device)
        old_log_probs = torch.stack(old_log_probs).to(device); rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device); values = torch.stack(values).to(device)
        with torch.no_grad():
            last = torch.as_tensor(model.flatten_observation(observation), dtype=torch.float32, device=device).unsqueeze(0)
            next_value = model.value(last)[0]

        advantages = torch.zeros_like(rewards); gae = torch.zeros((), device=device)
        for t in reversed(range(args.horizon)):
            nonterminal = 1.0 - dones[t]; next_v = next_value if t == args.horizon - 1 else values[t + 1]
            gae = rewards[t] + args.gamma * nonterminal * next_v - values[t] + args.gamma * args.gae_lambda * nonterminal * gae
            advantages[t] = gae

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(args.epochs):
            for indices in torch.randperm(args.horizon, device=device).split(args.batch_size):
                _, log_prob, value, _ = model.action_and_value(states[indices], raw_actions[indices])
                ratio = (log_prob - old_log_probs[indices]).exp(); clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                loss = -torch.min(ratio * advantages[indices], clipped * advantages[indices]).mean() + args.value_coef * 0.5 * (returns[indices] - value).square().mean()
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

        mean_reward = float(rewards.mean().item())
        log_writer.writerow({"update": update, "mean_reward": mean_reward, "distance_reward": float(sum(distance_rewards) / len(distance_rewards)), "fallen_reward": float(sum(fallen_rewards) / len(fallen_rewards)), "open_close_reward": float(sum(open_close_rewards) / len(open_close_rewards)), "mean_pin_distance": float(sum(pin_distances) / len(pin_distances)), "max_fallen_pins": max(fallen_counts)})
        log_file.flush()
        if (update + 1) % args.print_every == 0 or update == args.updates - 1:
            print(f"[POLICY UPDATED] update={update + 1} mean_reward={mean_reward:.3f} mean_distance={sum(distance_rewards) / len(distance_rewards):.3f} newly_fallen={sum(fallen_rewards):.0f}", flush=True)
            model.save(args.checkpoint)
    log_file.close()
    env.close(); model.save(args.checkpoint)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--updates", type=int, default=1000); p.add_argument("--horizon", type=int, default=512); p.add_argument("--episode-max-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128); p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4); p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--gae-lambda", type=float, default=.95); p.add_argument("--clip", type=float, default=.2)
    p.add_argument("--value-coef", type=float, default=.5); p.add_argument("--checkpoint", default="pincer_mlp.pt")
    p.add_argument("--device", default="cuda"); p.add_argument("--seed", type=int, default=0); p.add_argument("--print-every", type=int, default=10); p.add_argument("--render", action="store_true", help="show the MuJoCo viewer during training"); p.add_argument("--log-actions", action="store_true", help="periodically print compact action and pincer state summaries"); p.add_argument("--log-every", type=int, default=500, help="simulator steps between optional action logs"); p.add_argument("--log-file", default="training_rewards.csv", help="CSV file for reward metrics")
    train(p.parse_args())

if __name__ == "__main__":
    main()
