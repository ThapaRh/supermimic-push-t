"""
Evaluate a v3 policy in the REAL simulator and draw episodes.

    python3 eval_v3.py --ckpt policy_v3.pt --episodes 20 --plot eval_v3.png
"""

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle

from pusht_env import PushT, MAX_STEPS, GOAL


def mlp(i, o, h=256, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def obs(s):  # identical to supermimic_v3.obs
    c, sn = torch.cos(s[..., 4]), torch.sin(s[..., 4])
    dx, dy = s[..., 0] - s[..., 2], s[..., 1] - s[..., 3]
    rx, ry = c * dx + sn * dy, -sn * dx + c * dy
    return torch.stack([s[..., 0] / 256 - 1, s[..., 1] / 256 - 1,
                        s[..., 2] / 256 - 1, s[..., 3] / 256 - 1,
                        c, sn, rx / 100, ry / 100,
                        s[..., 5] / 100, s[..., 6] / 100, s[..., 7]], -1)


def t_polys(x, y, th):
    c, s = np.cos(th), np.sin(th)
    out = []
    for (w, h, oy) in [(120, 30, 45), (30, 90, -15)]:
        pts = np.array([[-w / 2, oy - h / 2], [w / 2, oy - h / 2], [w / 2, oy + h / 2], [-w / 2, oy + h / 2]])
        out.append(np.stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]], 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="policy_v3.pt")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--plot", default="eval_v3.png")
    ap.add_argument("--n_plot", type=int, default=6)
    ap.add_argument("--seed0", type=int, default=1000, help="1000-1019 were used for checkpoint selection; use 2000+ for held-out")
    ap.add_argument("--random_starts", action="store_true", help="T anywhere, any angle (harder than the training eval)")
    a = ap.parse_args()
    policy = mlp(11, 2)
    policy.load_state_dict(torch.load(a.ckpt)["policy"])
    rows, trajs = [], []
    with torch.no_grad():
        for k in range(a.episodes):
            e = PushT(seed=a.seed0 + k)
            if a.random_starts:
                import random
                e.reset_random(random.Random(a.seed0 + k))
            start = e.raw().copy()
            path = [start.copy()]
            ok = False
            for t in range(MAX_STEPS):
                act = torch.tanh(policy(obs(torch.tensor(e.raw()).unsqueeze(0))))[0].numpy()
                path.append(e.step(act).copy())
                if e.success():
                    ok = True
                    break
            pe, ae = e.errors()
            rows.append(dict(seed=a.seed0 + k, success=ok, pos_err=round(pe, 1), ang_err=round(ae, 3), steps=t + 1))
            trajs.append(np.array(path))
            print(rows[-1])
    sr = np.mean([r["success"] for r in rows])
    print(f"success {sr * 100:.0f}%  mean pos err {np.mean([r['pos_err'] for r in rows]):.1f} px")
    json.dump(rows, open(a.plot.replace(".png", ".json"), "w"), indent=1)

    n = min(a.n_plot, len(trajs))
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.4))
    for ax, tr, r in zip(np.atleast_1d(axs), trajs, rows):
        for poly in t_polys(*GOAL):
            ax.add_patch(Polygon(poly, fc="#b7e4b7", ec="g", ls="--"))
        for poly in t_polys(tr[0, 2], tr[0, 3], tr[0, 4]):
            ax.add_patch(Polygon(poly, fc="none", ec="0.6"))
        for poly in t_polys(tr[-1, 2], tr[-1, 3], tr[-1, 4]):
            ax.add_patch(Polygon(poly, fc="0.5", ec="k", alpha=0.8))
        ax.plot(tr[:, 0], tr[:, 1], "b-", lw=1)
        ax.add_patch(Circle(tr[-1, :2], 20, fc="#8ecae6", ec="b"))
        ax.set_xlim(0, 512), ax.set_ylim(0, 512), ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"seed {r['seed']}: {'OK' if r['success'] else 'fail'} "
                     f"{r['pos_err']:.0f}px {r['ang_err']:.2f}rad", fontsize=8,
                     color="g" if r["success"] else "r")
    plt.tight_layout()
    plt.savefig(a.plot, dpi=120)
    print("saved", a.plot)


if __name__ == "__main__":
    main()
