"""
supermimic-push-t -- Step 1: fit the world model, check it honestly.

This is SuperMimic's step (2) in the paper, shrunk to Push-T:
  1. Collect transitions (s, a, r, s_next) from the real simulator (pymunk).
  2. Fit ONE differentiable model f(s, a) -> (s_next, reward, done).
     The model is a small MLP. It is trained on 8-step windows,
     rolled forward on its own predictions, like Eq. (1) in the paper.
  3. Report 1-step and 8-step prediction error on held-out episodes.
     If the 8-step error is bad, do NOT train a policy through it yet.
     A policy trained through a bad model just learns to exploit the model.

State (15 numbers, all roughly in [-1, 1] after scaling):
  pusher x, y | T-block x, y, cos(angle), sin(angle) |
  pusher vx, vy | T-block vx, vy, angular velocity |
  target x, y, cos(target angle), sin(target angle)

Action (2 numbers): pusher velocity, scaled by 150 px/s.
Reward: -(position error / 512 + 0.5 * angle error). 0 is perfect, negative is worse.
Done: 1 if the T-block is close enough to the target, else 0.

On your Mac:
    source .venv/bin/activate
    pip install torch pymunk matplotlib numpy
    python train_world_model.py --episodes 100 --epochs 10
    # full run (slower, CPU): --episodes 200 --epochs 20

Outputs: world_model.pt, world_model_metrics.json, world_model_loss.png
"""

import argparse
import json
import math
import random

import numpy as np
import pymunk

# torch is imported lazily in main() so --collect-only can run without it.
STATE_DIM = 15
ACTION_DIM = 2


def angle_diff(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def add_t_block(space, x, y, angle):
    body = pymunk.Body(1, pymunk.moment_for_box(1, (120, 120)))
    body.position = (x, y)
    body.angle = angle
    top = pymunk.Poly.create_box(body, (120, 30))
    top.offset = (0, 45)
    top.friction = 0.5
    stem = pymunk.Poly.create_box(body, (30, 90))
    stem.offset = (0, -15)
    stem.friction = 0.5
    space.add(body, top, stem)
    return body


class PushT:
    """One Push-T episode in pymunk. Same table as push_t_random.py."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        for a, b in [((0, 0), (512, 0)), ((512, 0), (512, 512)),
                     ((512, 512), (0, 512)), ((0, 512), (0, 0))]:
            wall = pymunk.Segment(self.space.static_body, a, b, 1)
            wall.friction = 0.5
            self.space.add(wall)
        self.target = (256.0, 256.0, 0.0)
        self.t_body = add_t_block(
            self.space,
            150 + self.rng.randint(-30, 30),
            150 + self.rng.randint(-30, 30),
            self.rng.uniform(-0.5, 0.5))
        self.pusher = pymunk.Body(1, pymunk.moment_for_circle(1, 0, 20))
        self.pusher.position = (380 + self.rng.randint(-20, 20),
                                380 + self.rng.randint(-20, 20))
        shape = pymunk.Circle(self.pusher, 20)
        shape.friction = 0.5
        self.space.add(self.pusher, shape)

    def state(self):
        tx, ty, ta = self.target
        return np.array([
            self.pusher.position.x / 512.0,
            self.pusher.position.y / 512.0,
            self.t_body.position.x / 512.0,
            self.t_body.position.y / 512.0,
            math.cos(self.t_body.angle),
            math.sin(self.t_body.angle),
            self.pusher.velocity.x / 150.0,
            self.pusher.velocity.y / 150.0,
            self.t_body.velocity.x / 150.0,
            self.t_body.velocity.y / 150.0,
            self.t_body.angular_velocity / 5.0,
            tx / 512.0, ty / 512.0,
            math.cos(ta), math.sin(ta),
        ], dtype=np.float32)

    def reward_done(self):
        tx, ty, ta = self.target
        pos_err = math.dist(self.t_body.position, (tx, ty))
        ang_err = angle_diff(self.t_body.angle, ta)
        reward = -(pos_err / 512.0 + 0.5 * ang_err)
        done = 1.0 if (pos_err < 30 and ang_err < 0.3) else 0.0
        return reward, done

    def step(self, action):
        # action is (vx, vy) in [-1, 1]; scale to px/s.
        self.pusher.velocity = (float(action[0]) * 150.0,
                                float(action[1]) * 150.0)
        self.space.step(1 / 60.0)
        return self.state(), *self.reward_done()


def collect_episode(seed):
    """Run one random episode. Returns states, actions, rewards, dones."""
    env = PushT(seed=seed)
    states = [env.state()]
    actions, rewards, dones = [], [], []
    action = np.zeros(2, dtype=np.float32)
    for step in range(300):
        if step % 20 == 0:
            action = np.array([random.uniform(-1, 1),
                               random.uniform(-1, 1)], dtype=np.float32)
        s_next, r, done = env.step(action)
        states.append(s_next)
        actions.append(action.copy())
        rewards.append(r)
        dones.append(done)
        if done:
            break
    return (np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.float32))


def collect(n_episodes, seed0=0):
    eps = [collect_episode(seed0 + i) for i in range(n_episodes)]
    print(f"Collected {n_episodes} episodes, "
          f"{sum(len(e[1]) for e in eps)} transitions")
    return eps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--k", type=int, default=8,
                        help="world-model window length (paper: 8)")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    episodes = collect(args.episodes)
    n_train = int(0.8 * len(episodes))
    train_eps, val_eps = episodes[:n_train], episodes[n_train:]
    print(f"Train episodes: {len(train_eps)}, held-out: {len(val_eps)}")
    if args.collect_only:
        print("collect-only: stopping before torch training.")
        return

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

    model = WorldModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    # Flatten train episodes into (s, a, r, done, s_next) for window starts.
    # A window is k consecutive steps starting at one state.
    def sample_windows(eps, batch=128):
        # Pick a random episode, then a random start that leaves k steps.
        s0, a_win, s_true, r_true, d_true = [], [], [], [], []
        for _ in range(batch):
            ep_s, ep_a, ep_r, ep_d = random.choice(eps)
            start = random.randint(0, len(ep_a) - args.k)
            s0.append(ep_s[start])
            a_win.append(ep_a[start:start + args.k])
            s_true.append(ep_s[start + 1:start + args.k + 1])
            r_true.append(ep_r[start:start + args.k])
            d_true.append(ep_d[start:start + args.k])
        return (torch.tensor(np.array(s0), device=device),
                torch.tensor(np.array(a_win), device=device),
                torch.tensor(np.array(s_true), device=device),
                torch.tensor(np.array(r_true), device=device),
                torch.tensor(np.array(d_true), device=device))

    losses = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for _ in range(50):  # 50 batches per epoch
            s, a_win, s_true, r_true, d_true = sample_windows(train_eps)
            s_pred = s
            loss = 0.0
            for j in range(args.k):
                s_next, r_pred, d_logit = model(s_pred, a_win[:, j])
                loss = loss + mse(s_next, s_true[:, j]) \
                    + mse(r_pred, r_true[:, j]) \
                    + bce(d_logit, d_true[:, j])
                s_pred = s_next  # feed the model's own prediction back in
            loss = loss / args.k
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss)
        losses.append(total / 50)
        print(f"epoch {epoch + 1}/{args.epochs}  loss {losses[-1]:.4f}")

    torch.save(model.state_dict(), "world_model.pt")

    # --- Honest evaluation on held-out episodes ---
    model.eval()
    with torch.no_grad():
        # 1-step error, in pixels for the T-block position.
        errs_1, errs_8 = [], []
        for ep_s, ep_a, _, _ in val_eps:
            s = torch.tensor(ep_s[:-1], device=device)
            a = torch.tensor(ep_a, device=device)
            s_next, _, _ = model(s, a)
            err = (s_next[:, 2:4] - torch.tensor(ep_s[1:, 2:4],
                                                 device=device)).abs().mean().item()
            errs_1.append(err * 512.0)
            # 8-step: start true, then roll the model forward on logged actions.
            start = random.randint(0, len(ep_a) - args.k)
            s_pred = torch.tensor(ep_s[start:start + 1], device=device)
            for j in range(args.k):
                s_pred, _, _ = model(
                    s_pred, torch.tensor(ep_a[start + j:start + j + 1],
                                         device=device))
            err8 = (s_pred[0, 2:4] - torch.tensor(
                ep_s[start + args.k, 2:4], device=device)).abs().mean().item()
            errs_8.append(err8 * 512.0)
        metrics = {
            "train_episodes": len(train_eps),
            "val_episodes": len(val_eps),
            "final_loss": losses[-1],
            "one_step_T_pos_error_px_mean": float(np.mean(errs_1)),
            "eight_step_T_pos_error_px_mean": float(np.mean(errs_8)),
            "k": args.k,
        }
    print("Metrics:", json.dumps(metrics, indent=2))
    with open("world_model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(losses)
        plt.xlabel("epoch")
        plt.ylabel("world-model loss")
        plt.savefig("world_model_loss.png")
        print("Saved world_model_loss.png")
    except Exception as e:
        print(f"Skipped loss plot ({e})")


if __name__ == "__main__":
    main()
