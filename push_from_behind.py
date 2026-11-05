"""
Push-from-behind test — isolate pushing from approach.

Place the pusher already behind the T, on the opposite side
from the target, then push straight toward the target for
300 steps. No learning, no navigation, just physics.

If this succeeds, pushing works and approach (getting behind
the T without pushing it the wrong way first) is the hard part.
If this also fails, the pushing physics itself is the problem.

Usage, same folder, venv active:
    python3 push_from_behind.py
"""

import json
import math

import numpy as np

from train_world_model import PushT


def main():
    successes, pos_errs, ang_errs = [], [], []
    for seed in range(20):
        env = PushT(seed=1000 + seed)
        tx, ty = env.t_body.position
        gx, gy = 256.0, 256.0
        dx, dy = gx - tx, gy - ty
        dist = math.hypot(dx, dy) or 1.0
        ux, uy = dx / dist, dy / dist
        # Place pusher 50 px behind the T, stop everything.
        env.pusher.position = (tx - ux * 50.0, ty - uy * 50.0)
        env.pusher.velocity = (0, 0)
        env.t_body.velocity = (0, 0)
        env.t_body.angular_velocity = 0
        action = np.array([ux, uy], dtype=np.float32)
        done = False
        for _ in range(300):
            _, _, done = env.step(action)
            if done:
                break
        pos_err = math.dist(env.t_body.position, (gx, gy))
        ang_err = abs((env.t_body.angle - 0.0 + math.pi) %
                      (2 * math.pi) - math.pi)
        pos_errs.append(pos_err)
        ang_errs.append(ang_err)
        successes.append(1.0 if done else 0.0)
        print(f"seed {1000 + seed}: success {bool(done)} "
              f"pos err {pos_err:.0f} px  angle err {ang_err:.2f} rad  "
              f"T final ({env.t_body.position.x:.0f}, {env.t_body.position.y:.0f})")
    metrics = {
        "policy": "push straight from behind, no approach",
        "real_success_rate_20_eps": float(np.mean(successes)),
        "real_final_pos_error_px_mean": float(np.mean(pos_errs)),
        "real_final_angle_error_rad_mean": float(np.mean(ang_errs)),
    }
    print("Metrics:", json.dumps(metrics, indent=2))
    with open("push_from_behind_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
