"""
Heuristic v2 — go AROUND the T, then push.

v1 failed because it went straight from the pusher start to the
behind spot, straight through the T, pushing it southwest away
from the target.

v2 adds a waypoint, in plain terms:
  1. Go to a waypoint southeast of the T, (T_x + 120, T_y - 120),
     a point that is clear of the T. This goes around the south
     side, not through the T.
  2. From there, go to the behind spot, (T - 40 px toward target).
     This approach is from the south, so the pusher slides in
     behind the T without hitting it from the front.
  3. Once within 40 px of the T, push straight toward the target.

Usage, same folder, venv active:
    python3 heuristic_v2.py
"""

import json
import math

import numpy as np

from train_world_model import PushT


def unit_to(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d


def heuristic_v2_action(env):
    px, py = env.pusher.position
    tx, ty = env.t_body.position
    gx, gy = 256.0, 256.0
    ux, uy = unit_to(tx, ty, gx, gy)
    # Behind spot, 40 px behind the T.
    bx, by = tx - ux * 40.0, ty - uy * 40.0
    # Waypoint southeast of the T, to go around the south side.
    wx, wy = tx + 120.0, ty - 120.0
    d_pt = math.dist((px, py), (tx, ty))
    d_pw = math.dist((px, py), (wx, wy))
    if d_pw > 30 and d_pt > 80:
        # Stage 1: go to the waypoint, unless we are already close to the T.
        vx, vy = unit_to(px, py, wx, wy)
    elif d_pt > 40:
        # Stage 2: go to the behind spot.
        vx, vy = unit_to(px, py, bx, by)
    else:
        # Stage 3: push toward the target.
        vx, vy = ux, uy
    return np.array([vx, vy], dtype=np.float32)


def main():
    successes, pos_errs, ang_errs = [], [], []
    for seed in range(20):
        env = PushT(seed=1000 + seed)
        done = False
        for _ in range(300):
            a = heuristic_v2_action(env)
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
        "policy": "heuristic v2 (waypoint around south side, then push)",
        "real_success_rate_20_eps": float(np.mean(successes)),
        "real_final_pos_error_px_mean": float(np.mean(pos_errs)),
        "real_final_angle_error_rad_mean": float(np.mean(ang_errs)),
    }
    print("Metrics:", json.dumps(metrics, indent=2))
    with open("heuristic_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
