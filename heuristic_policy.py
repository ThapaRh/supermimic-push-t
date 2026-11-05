"""
Heuristic Push-T policy — a sanity-check baseline, NOT learned.

Two stages, in plain terms:
  1. If the pusher is far from the T (> 40 px), go to a spot just
     behind the T, on the opposite side from the target.
  2. Once behind the T, push straight toward the target.

This is the "does the environment even allow a solution, and what
does a working push look like" check. If this heuristic gets some
successes, the env is solvable and the learned policies are failing
for a learning reason, not an env reason. If it also gets 0%, the
env / physics setup itself is the problem.

Usage, same folder, venv active:
    python3 heuristic_policy.py

Outputs: heuristic_metrics.json, success rate over 20 episodes.
"""

import json
import math

import numpy as np

from train_world_model import PushT


def heuristic_action(env):
    px, py = env.pusher.position
    tx, ty = env.t_body.position
    gx, gy = 256.0, 256.0
    # Direction from T to target.
    dx, dy = gx - tx, gy - ty
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    # Desired pusher spot: 40 px behind the T, away from the target.
    desired = (tx - ux * 40.0, ty - uy * 40.0)
    d_pt = math.dist((px, py), (tx, ty))
    if d_pt > 40:
        # Stage 1: go to the behind-T spot.
        vx, vy = desired[0] - px, desired[1] - py
    else:
        # Stage 2: push toward the target.
        vx, vy = ux, uy
    norm = math.hypot(vx, vy) or 1.0
    return np.array([vx / norm, vy / norm], dtype=np.float32)


def main():
    successes, pos_errs, ang_errs = [], [], []
    for seed in range(20):
        env = PushT(seed=1000 + seed)
        done = False
        for _ in range(300):
            a = heuristic_action(env)
            _, _, done = env.step(a)
            if done:
                break
        tx, ty, ta = env.target
        pos_err = math.dist(env.t_body.position, (tx, ty))
        ang_err = abs((env.t_body.angle - ta + math.pi) %
                      (2 * math.pi) - math.pi)
        pos_errs.append(pos_err)
        ang_errs.append(ang_err)
        successes.append(1.0 if done else 0.0)
        print(f"seed {1000 + seed}: success {bool(done)} "
              f"pos err {pos_err:.0f} px  angle err {ang_err:.2f} rad")
    metrics = {
        "policy": "heuristic (approach behind T, then push to target)",
        "real_success_rate_20_eps": float(np.mean(successes)),
        "real_final_pos_error_px_mean": float(np.mean(pos_errs)),
        "real_final_angle_error_rad_mean": float(np.mean(ang_errs)),
    }
    print("Metrics:", json.dumps(metrics, indent=2))
    with open("heuristic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
