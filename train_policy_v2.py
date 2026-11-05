"""
supermimic-push-t -- Step 2, v2: policy through the frozen world model,
with a fix for why v1 did not learn.

v1 result (Nov 5, 200 iters): imagined return -11.9 -> -15.7 (worse),
real success 0/20, final pos error 152 px. Critic loss fell, policy did not.

Diagnosis, in plain terms:
  The reward only cares about where the T-block is. Until the pusher
  actually touches the T, the T does not move, so the reward does not
  change, so the policy gets no gradient telling it to go TO the T first.
  From a far start, 32 imagined steps is not enough to stumble into
  contact by chance, so the policy learned nothing useful, then drifted.

v2 changes, and why:
  1. Shaped training reward: model reward - 0.3 * (pusher-to-T distance).
     This gives a gradient to approach the T first. Evaluation is still
     on the ORIGINAL success metric only (pos < 30 px, angle < 0.3 rad),
     so the shaping is a training aid, not a fudged score. Say so in the report.
  2. Near-contact starts: 80% of imagined rollouts start with the pusher
     placed 60-120 px from the T, so contact happens inside H = 32 steps
     and pushing gets a direct reward signal. 20% are far random starts,
     so approach is not forgotten entirely.
  3. Lower policy learning rate (1e-4, was 3e-4): v1 drifted worse, so
     take smaller steps.
  4. Logging: imagined T error at start vs end, and pusher-to-T distance,
     so you can see whether the policy is actually moving the T in
     imagination, before trusting the real-simulator eval.

On your Mac, same folder, venv active:
    python3 train_policy_v2.py --iters 300

Outputs: policy_v2.pt, policy_metrics_v2.json
"""

import argparse
import json
import math
import random

import numpy as np

from train_world_model import PushT, collect, STATE_DIM, ACTION_DIM


def near_start_states(n, rng):
    """Start states with the pusher already near the T-block."""
    states = []
    for _ in range(n):
        env = PushT(seed=rng.randint(0, 10**9))
        # Place the pusher 60-120 px from the T, random direction.
        ang = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(60, 120)
        env.pusher.position = (
            env.t_body.position.x + dist * math.cos(ang),
            env.t_body.position.y + dist * math.sin(ang),
        )
        env.pusher.velocity = (0, 0)
        env.t_body.velocity = (0, 0)
        env.t_body.angular_velocity = 0
        states.append(env.state())
    return np.array(states, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

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
    opt_p = torch.optim.Adam(policy.parameters(), lr=1e-4)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # Start pool: 80% near-contact, 20% far random states.
    rng = random.Random(0)
    near = near_start_states(4000, rng)
    eps = collect(50)
    far = np.concatenate([e[0][:-1] for e in eps], axis=0)
    starts = np.concatenate([near, far[:1000]], axis=0)
    print(f"Start-state pool: {len(starts)} states "
          f"({len(near)} near, 1000 far)")
    starts_t = torch.tensor(starts, device=device)

    def pusher_t_dist(s):
        # s is (..., 15): pusher x,y are [0:2], T x,y are [2:4], in /512 units.
        return torch.sqrt(((s[..., 0:2] - s[..., 2:4]) ** 2).sum(-1)) * 512.0

    def t_pos_error(s):
        # T x,y vs target 256,256.
        return torch.sqrt(((s[..., 2:4] * 512.0 - 256.0) ** 2).sum(-1))

    gamma, lam = 0.99, 0.95
    H = args.horizon
    returns_log = []

    for it in range(args.iters):
        idx = torch.randint(0, len(starts_t), (args.batch,), device=device)
        s = starts_t[idx]
        states, rewards, dones = [s], [], []
        for _ in range(H):
            a = policy(s)
            s_next, r_model, d_logit = world(s, a)
            s_next = torch.clamp(s_next, -1.5, 1.5)
            # Shaped reward: push the T (model reward) AND approach it.
            r = r_model - 0.3 * (pusher_t_dist(s_next) / 512.0)
            states.append(s_next)
            rewards.append(r)
            dones.append(torch.sigmoid(d_logit))
            s = s_next

        rewards_t = torch.stack(rewards, dim=1)
        dones_t = torch.stack(dones, dim=1)
        states_t = torch.stack(states, dim=1)
        values = critic(states_t.reshape(-1, STATE_DIM)).reshape(
            args.batch, H + 1)

        with torch.no_grad():
            v = values.detach()
            ret = torch.zeros(args.batch, device=device)
            rets = []
            for t in reversed(range(H)):
                alive = 1.0 - dones_t[:, t]
                ret = rewards_t[:, t] + gamma * alive * (
                    (1 - lam) * v[:, t + 1] + lam * ret)
                rets.append(ret)
            rets = torch.stack(rets[::-1], dim=1)

        survival = torch.cumprod(1.0 - dones_t, dim=1)
        discounts = torch.tensor(
            [gamma ** t for t in range(H)], device=device)
        weights = discounts.unsqueeze(0) * survival
        imagined_return = (weights * rewards_t).sum(dim=1) + \
            (gamma ** H) * survival[:, -1] * values[:, -1].detach()
        policy_loss = -imagined_return.mean()
        critic_loss = ((values[:, :-1] - rets) ** 2).mean()

        opt_p.zero_grad()
        opt_c.zero_grad()
        (policy_loss + critic_loss).backward()
        opt_p.step()
        opt_c.step()

        returns_log.append(float(imagined_return.mean().detach()))
        if (it + 1) % 20 == 0 or it == 0:
            with torch.no_grad():
                err_start = float(t_pos_error(states_t[:, 0]).mean())
                err_end = float(t_pos_error(states_t[:, -1]).mean())
                dist_end = float(pusher_t_dist(states_t[:, -1]).mean())
            print(f"iter {it + 1}/{args.iters}  "
                  f"imagined return {returns_log[-1]:.3f}  "
                  f"critic loss {float(critic_loss.detach()):.4f}  "
                  f"imagined T err {err_start:.0f} -> {err_end:.0f} px  "
                  f"pusher-T dist end {dist_end:.0f} px")

    torch.save(policy.state_dict(), "policy_v2.pt")

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
        "version": "v2 shaped + near starts",
        "iters": args.iters,
        "horizon": H,
        "imagined_return_final": returns_log[-1],
        "imagined_return_first": returns_log[0],
        "real_success_rate_20_eps": float(np.mean(successes)),
        "real_final_pos_error_px_mean": float(np.mean(pos_errs)),
        "real_final_angle_error_rad_mean": float(np.mean(ang_errs)),
    }
    print("Metrics:", json.dumps(metrics, indent=2))
    with open("policy_metrics_v2.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved policy_v2.pt, policy_metrics_v2.json")


if __name__ == "__main__":
    main()
