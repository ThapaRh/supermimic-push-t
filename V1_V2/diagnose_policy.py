"""
Diagnose a trained policy in the REAL simulator.

Usage, same folder, venv active:
    python3 diagnose_policy.py --policy policy_v2.pt
    python3 diagnose_policy.py --policy policy.pt

Prints, every 50 steps, for 3 real episodes:
  pusher position, T position, action the policy chose,
  pusher-to-T distance, T-to-target distance.
If the action is ~0, or the pusher never gets near the T, or the
T never moves, the policy is inert — that is the diagnosis, not a bug
in the printout.
"""

import argparse
import math

import numpy as np

from train_world_model import PushT, STATE_DIM, ACTION_DIM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="policy_v2.pt")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    import torch
    import torch.nn as nn

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(STATE_DIM, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, ACTION_DIM),
            )

        def forward(self, s):
            return torch.tanh(self.net(s))

    policy = Policy().to(device)
    policy.load_state_dict(torch.load(args.policy, map_location=device))
    policy.eval()
    print(f"Loaded {args.policy}")

    with torch.no_grad():
        for ep in range(args.episodes):
            env = PushT(seed=1000 + ep)
            s = env.state()
            print(f"\nEpisode {ep}: T start {env.t_body.position}, "
                  f"pusher start {env.pusher.position}")
            for step in range(300):
                a = policy(torch.tensor(s, device=device).unsqueeze(0))
                a_np = a.cpu().numpy()[0]
                s, r, done = env.step(a_np)
                if step % 50 == 0 or done:
                    d_pt = math.dist(env.pusher.position, env.t_body.position)
                    d_tt = math.dist(env.t_body.position, (256, 256))
                    print(f"  step {step:3d} action ({a_np[0]:+.2f}, {a_np[1]:+.2f}) "
                          f"pusher ({env.pusher.position.x:.0f}, {env.pusher.position.y:.0f}) "
                          f"T ({env.t_body.position.x:.0f}, {env.t_body.position.y:.0f}) "
                          f"pusher-T {d_pt:.0f} px  T-target {d_tt:.0f} px")
                if done:
                    print("  SUCCESS")
                    break
            else:
                print("  end: no success, T never reached target")


if __name__ == "__main__":
    main()
