"""
supermimic-push-t -- Step 2: train a policy through the frozen world model.

This is SuperMimic's step (3) in the paper, shrunk to Push-T:
  1. Load the world model from Step 1 (world_model.pt) and FREEZE it.
  2. Imagine short rollouts, H = 32 steps, starting from states collected
     in the real simulator. The policy acts, the frozen model predicts
     the next state and reward. Gradients flow back through the model
     to the policy (backpropagation through time, like SHAC / Eq. 2).
  3. A critic V(s) bootstraps the return after H steps, so the policy
     does not need to imagine all 300 steps. The critic is fitted to
     lambda-returns computed in imagination (paper Eq. 3, lambda = 0.95).
  4. Judge ONLY in the real simulator: run the policy in pymunk over
     varied starts and report success rate. Imagined reward is not a result.

On your Mac, in the same folder, venv active:
    python train_policy.py --iters 200
    # outputs: policy.pt, policy_metrics.json

The world model is honest but imperfect (1-step error ~15 px), so
expect the policy to be imperfect too. Success > random (0%) in the
real simulator is the first real signal.
"""

import argparse
import json
import math
import random

import numpy as np

from train_world_model import PushT, collect, STATE_DIM, ACTION_DIM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=32,
                        help="imagined horizon H (paper: 32)")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--collect-episodes", type=int, default=50,
                        help="random episodes to harvest start states from")
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    # --- Load and freeze the world model (same architecture as Step 1) ---
    class WorldModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM + ACTION_DIM, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, STATE_DIM + 2),
            )

        def forward(self, s, a):
            out = self.net(torch.cat([s, a], dim=-1))
            return out[:, :STATE_DIM], out[:, STATE_DIM], out[:, STATE_DIM + 1]

    world = WorldModel().to(device)
    world.load_state_dict(torch.load("world_model.pt", map_location=device))
    world.eval()
    for p in world.parameters():
        p.requires_grad = False
    print("Loaded world_model.pt, frozen.")

    # --- Policy and critic ---
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, ACTION_DIM),
            )

        def forward(self, s):
            return torch.tanh(self.net(s))

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, args.hidden),
                nn.ReLU(),
                nn.Linear(args.hidden, 1),
            )

        def forward(self, s):
            return self.net(s).squeeze(-1)

    policy = Policy().to(device)
    critic = Critic().to(device)
    opt_p = torch.optim.Adam(policy.parameters(), lr=3e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # --- Start states from the real simulator ---
    eps = collect(args.collect_episodes)
    starts = np.concatenate([e[0][:-1] for e in eps], axis=0)
    print(f"Start-state pool: {len(starts)} states from "
          f"{args.collect_episodes} episodes")
    starts_t = torch.tensor(starts, device=device)

    gamma, lam = 0.99, 0.95
    H = args.horizon
    returns_log = []

    for it in range(args.iters):
        idx = torch.randint(0, len(starts_t), (args.batch,), device=device)
        s = starts_t[idx]
        states, rewards, dones = [s], [], []
        for _ in range(H):
            a = policy(s)
            s_next, r, d_logit = world(s, a)
            # Keep imagined states in a sane range; the model can drift.
            s_next = torch.clamp(s_next, -1.5, 1.5)
            states.append(s_next)
            rewards.append(r)
            dones.append(torch.sigmoid(d_logit))
            s = s_next

        rewards_t = torch.stack(rewards, dim=1)          # (B, H)
        dones_t = torch.stack(dones, dim=1)              # (B, H)
        states_t = torch.stack(states, dim=1)            # (B, H+1, 15)
        values = critic(states_t.reshape(-1, STATE_DIM)).reshape(
            args.batch, H + 1)

        # Lambda-returns, computed with the critic detached (targets).
        with torch.no_grad():
            v = values.detach()
            ret = torch.zeros(args.batch, device=device)
            rets = []
            for t in reversed(range(H)):
                alive = 1.0 - dones_t[:, t]
                ret = rewards_t[:, t] + gamma * alive * (
                    (1 - lam) * v[:, t + 1] + lam * ret)
                rets.append(ret)
            rets = torch.stack(rets[::-1], dim=1)  # (B, H)

        # Policy objective: imagined return + critic bootstrap at H.
        # Survival weighting: stop counting reward after predicted done.
        survival = torch.cumprod(1.0 - dones_t, dim=1)
        discounts = torch.tensor(
            [gamma ** t for t in range(H)], device=device)
        weights = discounts.unsqueeze(0) * survival
        imagined_return = (weights * rewards_t).sum(dim=1) + \
            (gamma ** H) * survival[:, -1] * values[:, -1].detach()
        policy_loss = -imagined_return.mean()

        # Critic fit: predict the lambda-returns at each imagined state.
        critic_loss = ((values[:, :-1] - rets) ** 2).mean()

        # One backward for both: policy_loss and critic_loss share the
        # critic forward pass, so a second backward() would fail with
        # "Trying to backward through the graph a second time".
        # The bootstrap value in imagined_return is detached, so the
        # policy loss does not push the critic around.
        opt_p.zero_grad()
        opt_c.zero_grad()
        (policy_loss + critic_loss).backward()
        opt_p.step()
        opt_c.step()

        returns_log.append(float(imagined_return.mean().detach()))
        if (it + 1) % 20 == 0 or it == 0:
            print(f"iter {it + 1}/{args.iters}  "
                  f"imagined return {returns_log[-1]:.3f}  "
                  f"critic loss {float(critic_loss.detach()):.4f}")

    torch.save(policy.state_dict(), "policy.pt")

    # --- Honest evaluation: real simulator only ---
    print("Evaluating in the real simulator...")
    policy.eval()
    successes, pos_errs, ang_errs = [], [], []
    with torch.no_grad():
        for seed in range(20):
            env = PushT(seed=1000 + seed)
            s = env.state()
            done = False
            for _ in range(300):
                a = policy(torch.tensor(s, device=device).unsqueeze(0))
                s, r, done = env.step(a.cpu().numpy()[0])
                if done:
                    break
            tx, ty, ta = env.target
            pos_err = math.dist(env.t_body.position, (tx, ty))
            ang_err = abs((env.t_body.angle - ta + math.pi) %
                          (2 * math.pi) - math.pi)
            pos_errs.append(pos_err)
            ang_errs.append(ang_err)
            successes.append(1.0 if done else 0.0)

    metrics = {
        "iters": args.iters,
        "horizon": H,
        "imagined_return_final": returns_log[-1],
        "imagined_return_first": returns_log[0],
        "real_success_rate_20_eps": float(np.mean(successes)),
        "real_final_pos_error_px_mean": float(np.mean(pos_errs)),
        "real_final_angle_error_rad_mean": float(np.mean(ang_errs)),
    }
    print("Metrics:", json.dumps(metrics, indent=2))
    with open("policy_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved policy.pt, policy_metrics.json")


if __name__ == "__main__":
    main()
