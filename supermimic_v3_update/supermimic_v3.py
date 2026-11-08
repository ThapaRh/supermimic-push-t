"""
supermimic-push-t v3: the SuperMimic loop as the paper (and zita-ch's
supermimic_vibed) actually runs it.

What changed vs v1/v2, and why (see EXPERIMENTS.md, Step 3):
  * Env v2 (pusht_env.py): real T shape, damping, 10 Hz action repeat.
  * World model predicts INCREMENTS in the T's own frame, with running
    input/output normalisation and a zero-initialised last layer, so it
    starts as "nothing changes" (supermimic_vibed networks.py).
  * Reward is computed analytically from the predicted state, not learned.
  * Online loop, every iteration (paper Alg. A.1):
        collect 1 step in N envs (policy + action noise) -> replay buffer
        fit world model on k = 8 step windows
        policy BPTT through the frozen model, H = 32, critic bootstrap
        critic on TD(lambda) targets of the imagined rollout
  * Gradient clipping: norm 1.0, then element-wise 0.1 (policy).
  * World-model check against the "nothing changes" baseline, on contact
    transitions only.

Run:  python3 supermimic_v3.py --iters 6000
"""

import argparse
import json
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn

from pusht_env import PushT, GOAL, MAX_STEPS, SPEED, SUBSTEPS, DT

p = argparse.ArgumentParser()
p.add_argument("--iters", type=int, default=6000)
p.add_argument("--envs", type=int, default=32)
p.add_argument("--horizon", type=int, default=32)
p.add_argument("--k", type=int, default=8)
p.add_argument("--wm_batches", type=int, default=2)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--noise", type=float, default=0.3)
p.add_argument("--approach", type=float, default=0.5, help="weight of pusher-to-T shaping")
p.add_argument("--no_critic", action="store_true")
p.add_argument("--sparse", action="store_true", help="reward = success indicator (smoothed) only")
p.add_argument("--warmup", type=int, default=200, help="random-action collection steps per env")
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--pi_lr", type=float, default=1e-4)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--no_clamp", action="store_true", help="do not keep imagined states on the table")
p.add_argument("--tag", default="v3")
args = p.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
rng = random.Random(args.seed)
torch.set_num_threads(args.threads)
gamma, lam = 0.99, 0.95
GX, GY, GTH = GOAL
STEP_PX = SPEED * SUBSTEPS * DT  # max pusher travel per control step (30 px)


# ---------------------------------------------------------------- helpers
def rot(x, y, c, s):
    """Rotate vector (x, y) by angle with cos c, sin s."""
    return c * x - s * y, s * x + c * y


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.eps = eps

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1])
        m, v, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        tot = self.count + n
        d = m - self.mean
        self.mean += d * n / tot
        self.var = (self.var * self.count + v * n + d ** 2 * self.count * n / tot) / tot
        self.count = tot

    def std(self):
        return (self.var + self.eps ** 2).sqrt()

    def forward(self, x):
        return (x - self.mean) / self.std()


def mlp(i, o, h=256, n=3, ln=True):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h)] + ([nn.LayerNorm(h)] if ln else []) + [nn.SiLU()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


# State s (pixels/radians): [px, py, tx, ty, th, tvx, tvy, om]
def wm_inputs(s, a):
    c, sn = torch.cos(s[..., 4]), torch.sin(s[..., 4])
    rx, ry = rot(s[..., 0] - s[..., 2], s[..., 1] - s[..., 3], c, -sn)  # pusher in T frame
    ax, ay = rot(a[..., 0], a[..., 1], c, -sn)
    vx, vy = rot(s[..., 5], s[..., 6], c, -sn)
    # absolute positions only so the model can see walls
    return torch.stack([rx / 100, ry / 100, ax, ay, vx / 100, vy / 100, s[..., 7],
                        s[..., 0] / 256 - 1, s[..., 1] / 256 - 1,
                        s[..., 2] / 256 - 1, s[..., 3] / 256 - 1], -1)


def wm_targets(s, s1):
    """Increments in the T frame at time t."""
    c, sn = torch.cos(s[..., 4]), torch.sin(s[..., 4])
    dpx, dpy = rot(s1[..., 0] - s[..., 0], s1[..., 1] - s[..., 1], c, -sn)
    dtx, dty = rot(s1[..., 2] - s[..., 2], s1[..., 3] - s[..., 3], c, -sn)
    dth = torch.atan2(torch.sin(s1[..., 4] - s[..., 4]), torch.cos(s1[..., 4] - s[..., 4]))
    vx, vy = rot(s1[..., 5], s1[..., 6], c, -sn)
    return torch.stack([dpx, dpy, dtx, dty, dth, vx, vy, s1[..., 7]], -1)


class WorldModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(11, 8)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(11)
        self.out_norm = RunningNorm(8, eps=1e-3)

    def step(self, s, a):
        y = self.net(self.in_norm(wm_inputs(s, a)))
        d = y * self.out_norm.std() + self.out_norm.mean
        c, sn = torch.cos(s[..., 4]), torch.sin(s[..., 4])
        dpx, dpy = rot(d[..., 0], d[..., 1], c, sn)
        dtx, dty = rot(d[..., 2], d[..., 3], c, sn)
        vx, vy = rot(d[..., 5], d[..., 6], c, sn)
        th1 = s[..., 4] + d[..., 4]
        return torch.stack([s[..., 0] + dpx, s[..., 1] + dpy, s[..., 2] + dtx,
                            s[..., 3] + dty, th1, vx, vy, d[..., 7]], -1)


def obs(s):
    c, sn = torch.cos(s[..., 4]), torch.sin(s[..., 4])
    rx, ry = rot(s[..., 0] - s[..., 2], s[..., 1] - s[..., 3], c, -sn)
    return torch.stack([s[..., 0] / 256 - 1, s[..., 1] / 256 - 1,
                        s[..., 2] / 256 - 1, s[..., 3] / 256 - 1,
                        c, sn, rx / 100, ry / 100,
                        s[..., 5] / 100, s[..., 6] / 100, s[..., 7]], -1)


def reward(s1):
    """Analytic reward on the next state (same function in sim and model)."""
    pos = torch.sqrt((s1[..., 2] - GX) ** 2 + (s1[..., 3] - GY) ** 2 + 1e-6)
    ang = 1 - torch.cos(s1[..., 4] - GTH)
    d_pt = torch.sqrt((s1[..., 0] - s1[..., 2]) ** 2 + (s1[..., 1] - s1[..., 3]) ** 2 + 1e-6)
    approach = torch.relu(d_pt - 60) / 256
    if args.sparse:
        # smoothed success indicator: ~1 inside the success zone, ~0 outside
        succ = torch.sigmoid((30 - pos) / 5) * torch.sigmoid((0.045 - ang) / 0.01)
        return succ - 0.1 * args.approach * approach
    return -(pos / 256 + ang) - args.approach * approach


policy = mlp(11, 2, h=256, n=2)
critic = mlp(11, 1, h=256, n=2)
wm = WorldModel()
opt_wm = torch.optim.Adam(wm.net.parameters(), lr=1e-3)
opt_pi = torch.optim.Adam(policy.parameters(), lr=args.pi_lr, betas=(0.9, 0.95))
opt_v = torch.optim.Adam(critic.parameters(), lr=1e-3)


def act(s):
    return torch.tanh(policy(obs(s)))


# ---------------------------------------------------------------- replay buffer
CAP = args.warmup + args.iters + 1
S = np.zeros((args.envs, CAP, 8), np.float32)
A = np.zeros((args.envs, CAP, 2), np.float32)
S1 = np.zeros((args.envs, CAP, 8), np.float32)
EP = np.zeros((args.envs, CAP), np.int64)
n_buf = 0

envs, ep_id = [], np.zeros(args.envs, np.int64)
next_ep = 0


def new_env(i):
    global next_ep
    e = PushT(seed=rng.randint(0, 10 ** 9))
    if rng.random() < 0.7:
        e.reset_random(rng)
    ep_id[i] = next_ep
    next_ep += 1
    return e


envs = [new_env(i) for i in range(args.envs)]
rand_a = np.zeros((args.envs, 2), np.float32)


def collect(random_actions=False):
    global n_buf
    s_np = np.stack([e.raw() for e in envs])
    if random_actions:
        if n_buf % 5 == 0:
            rand_a[:] = np.random.uniform(-1, 1, rand_a.shape)
        a_np = rand_a.copy()
    else:
        with torch.no_grad():
            a_np = act(torch.tensor(s_np)).numpy()
        a_np = np.clip(a_np + args.noise * np.random.randn(*a_np.shape), -1, 1).astype(np.float32)
    for i, e in enumerate(envs):
        S1[i, n_buf] = e.step(a_np[i])
        S[i, n_buf], A[i, n_buf], EP[i, n_buf] = s_np[i], a_np[i], ep_id[i]
        if e.steps >= MAX_STEPS or e.success():
            envs[i] = new_env(i)
    st, s1t, at = torch.tensor(S[:, n_buf]), torch.tensor(S1[:, n_buf]), torch.tensor(A[:, n_buf])
    wm.in_norm.update(wm_inputs(st, at))
    wm.out_norm.update(wm_targets(st, s1t))
    critic_norm.update(obs(st))
    n_buf += 1


critic_norm = RunningNorm(11)


def sample_windows(B, k):
    ei = np.random.randint(0, args.envs, B * 2)
    ti = np.random.randint(0, n_buf - k + 1, B * 2)
    ok = EP[ei, ti] == EP[ei, ti + k - 1]
    ei, ti = ei[ok][:B], ti[ok][:B]
    idx = ti[:, None] + np.arange(k)[None]
    return (torch.tensor(S[ei, ti]), torch.tensor(A[ei[:, None], idx]),
            torch.tensor(S1[ei[:, None], idx]))


def state_err(sh, s):
    """Per-dim error in normalised increment units."""
    c, sn = torch.cos(s[..., 4]), torch.sin(s[..., 4])
    ex, ey = rot(sh[..., 0] - s[..., 0], sh[..., 1] - s[..., 1], c, -sn)
    tx, ty = rot(sh[..., 2] - s[..., 2], sh[..., 3] - s[..., 3], c, -sn)
    eth = torch.atan2(torch.sin(sh[..., 4] - s[..., 4]), torch.cos(sh[..., 4] - s[..., 4]))
    std = wm.out_norm.std()
    return torch.stack([ex / std[0], ey / std[1], tx / std[2], ty / std[3], eth / std[4],
                        (sh[..., 5] - s[..., 5]) / (std[5] + 1), (sh[..., 6] - s[..., 6]) / (std[6] + 1),
                        (sh[..., 7] - s[..., 7]) / (std[7] + 0.1)], -1)


def wm_update():
    s0, a, s1 = sample_windows(args.batch, args.k)
    sh, loss = s0, 0.0
    for j in range(args.k):
        sh = wm.step(sh, a[:, j])
        loss = loss + (state_err(sh, s1[:, j]) ** 2).mean()
    loss = loss / args.k
    opt_wm.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(wm.net.parameters(), 1.0)
    opt_wm.step()
    return float(loss.detach())


def V(s):
    return critic(critic_norm(obs(s))).squeeze(-1)


def policy_update():
    ei = np.random.randint(0, args.envs, args.batch)
    ti = np.random.randint(0, n_buf, args.batch)
    s = torch.tensor(S[ei, ti])
    for prm in wm.net.parameters():
        prm.requires_grad_(False)
    states, rews = [s], []
    for _ in range(args.horizon):
        s = wm.step(s, act(s))
        if not args.no_clamp:
            # keep imagined pusher/T on the table: the model never saw
            # states outside it, so extrapolating there blows up gradients
            s = torch.cat([s[..., :4].clamp(0, 512), s[..., 4:]], -1)
        states.append(s)
        rews.append(reward(s))
    for prm in wm.net.parameters():
        prm.requires_grad_(True)
    R = torch.stack(rews)  # (H, B)
    disc = gamma ** torch.arange(args.horizon, dtype=torch.float32).unsqueeze(-1)
    J = (disc * R).sum(0)
    if not args.no_critic:
        for prm in critic.parameters():
            prm.requires_grad_(False)
        J = J + gamma ** args.horizon * V(states[-1])
        for prm in critic.parameters():
            prm.requires_grad_(True)
    loss = -J.mean() / args.horizon
    opt_pi.zero_grad()
    loss.backward()
    gn = nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    nn.utils.clip_grad_value_(policy.parameters(), 0.1)
    if torch.isfinite(gn):
        opt_pi.step()
    vl = 0.0
    if not args.no_critic:
        with torch.no_grad():
            St = torch.stack([x.detach() for x in states])  # (H+1, B, 8)
            Vt = V(St)
            Rl, tg = Vt[-1], []
            for t in reversed(range(args.horizon)):
                Rl = R[t].detach() + gamma * ((1 - lam) * Vt[t + 1] + lam * Rl)
                tg.append(Rl)
            tg = torch.stack(tg[::-1])
        X, Y = St[:-1].reshape(-1, 8), tg.reshape(-1)
        for _ in range(2):
            perm = torch.randperm(len(X))
            for mb in perm.split(len(X) // 4):
                vl = ((V(X[mb]) - Y[mb]) ** 2).mean()
                opt_v.zero_grad()
                vl.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                opt_v.step()
        vl = float(vl)
    return float(-J.mean()), vl, float(gn)


@torch.no_grad()
def wm_check():
    """1-step T error on recent real transitions with contact, vs 'nothing changes'."""
    lo = max(0, n_buf - 300)
    s = torch.tensor(S[:, lo:n_buf].reshape(-1, 8))
    a = torch.tensor(A[:, lo:n_buf].reshape(-1, 2))
    s1 = torch.tensor(S1[:, lo:n_buf].reshape(-1, 8))
    moved = (s1[:, 2:4] - s[:, 2:4]).norm(dim=-1) > 0.5
    if moved.sum() < 10:
        return float("nan"), float("nan"), float(moved.float().mean())
    sh = wm.step(s[moved], a[moved])
    err = (sh[:, 2:4] - s1[moved, 2:4]).norm(dim=-1).mean()
    base = (s[moved, 2:4] - s1[moved, 2:4]).norm(dim=-1).mean()
    return float(err), float(base), float(moved.float().mean())


@torch.no_grad()
def evaluate(n=20):
    succ, pos_e, ang_e = [], [], []
    for k in range(n):
        e = PushT(seed=1000 + k)
        ok = False
        for _ in range(MAX_STEPS):
            e.step(act(torch.tensor(e.raw()).unsqueeze(0))[0].numpy())
            if e.success():
                ok = True
                break
        pe, ae = e.errors()
        succ.append(ok), pos_e.append(pe), ang_e.append(ae)
    return float(np.mean(succ)), float(np.mean(pos_e)), float(np.mean(ang_e))


# ---------------------------------------------------------------- main loop
t0 = time.time()
for _ in range(args.warmup):
    collect(random_actions=True)
for _ in range(300):  # world-model-only warm start
    wm_update()
log = []
best = (-1, None)
for it in range(1, args.iters + 1):
    # cosine decay of the policy learning rate to 0.1x
    for g in opt_pi.param_groups:
        g["lr"] = args.pi_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * it / args.iters)))
    collect()
    for _ in range(args.wm_batches):
        wl = wm_update()
    J, vl, gn = policy_update()
    if it % 100 == 0:
        e1, b1, frac = wm_check()
        print(f"it {it} | {time.time() - t0:.0f}s | wm {wl:.4f} | contact-step T err {e1:.2f} px "
              f"(no-change {b1:.2f}) contact frac {frac:.2f} | imag J {J:.3f} v {vl:.4f} gn {gn:.2f}",
              flush=True)
    if it % args.eval_every == 0 or it == args.iters:
        sr, pe, ae = evaluate()
        e1, b1, frac = wm_check()
        log.append(dict(it=it, success=sr, pos_err=pe, ang_err=ae, wm_contact_err=e1,
                        wm_nochange_err=b1, contact_frac=frac, sec=time.time() - t0))
        print(f"== EVAL it {it}: success {sr * 100:.0f}% | pos err {pe:.1f} px | ang err {ae:.2f} rad", flush=True)
        if sr >= best[0]:
            best = (sr, it)
            torch.save({"policy": policy.state_dict(), "wm": wm.state_dict(), "args": vars(args)},
                       f"policy_{args.tag}.pt")
json.dump({"args": vars(args), "best_success": best[0], "best_iter": best[1], "log": log},
          open(f"metrics_{args.tag}.json", "w"), indent=2)
print("done; best success", best)
