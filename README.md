# supermimic-push-t

Apply the SuperMimic framework to the standard Push-T task.

SuperMimic (Liao et al., arXiv 2610.28480, Oct 2026) trains a humanoid
to track motion-capture clips by learning one differentiable world
model and training the policy by backpropagation through short
imagined rollouts, with a critic bootstrapping the tail.

Here the same loop is transferred to Push-T, a 2D task where a round
pusher pushes a T-shaped block to a target pose:
- State: pusher position/velocity, T-block position, angle and velocity, target pose
- Action: pusher velocity (2 numbers)
- Reward: -(position error / 512 + 0.5 * angle error)
- Success: position error < 30 px and angle error < 0.3 rad

## Results so far

Step 0 — random pusher baseline (`push_t_random.py`):
- T-block final (174, 144), target (256, 256)
- Position error 139 px, angle error 0.26 rad, success: False

Step 1 — world model (`train_world_model.py`):
- 100 episodes / 30,000 transitions, train 80 / held-out 20
- Loss 0.4537 (epoch 1) → 0.0069 (epoch 10), k = 8-step windows
- 1-step T-position error: 15.3 px (held-out)
- 8-step T-position error: 17.7 px (held-out)

Step 2 v1 — policy through the frozen world model (`train_policy.py`):
- 200 iters, imagined return -11.9 -> -15.7 (worse), real success 0/20
- Real final position error 152.3 px, angle error 0.26 rad
- Did not learn to push; diagnosis and fix in `EXPERIMENTS.md`

Step 2 v2 — shaped reward + near-contact starts (`train_policy_v2.py`):
- 300 iters, imagined return -17.2 -> -22.2 (worse), real success 0/20
- Real final position error 152.3 px, angle error 0.26 rad, identical to v1
- Did not learn; T never moved in real eval. Diagnosis in `EXPERIMENTS.md`

Diagnostics — pushing works, approach is the hard part:
- Learned policy v2: pusher moves to a corner, away from the T, T never moves
- Naive heuristic (straight to behind spot): 0/20, 286 px, pushes T away on approach
- Push straight from behind, no approach: **12/20 (60%)**, successes at 28-30 px, failures spin/overshoot to the northeast corner
- Heuristic v2 (waypoint, stateless): 0/20, 152 px, T never moved — pusher oscillated at the waypoint
- Heuristic v3 (waypoint with latch, assistant VM test): 0/20, 237 px, pusher hit the T sideways and spun it away

Note: the Push-T setup used here (starts, target, success rule) is the standard public Push-T task, not a specification from Tang — see `EXPERIMENTS.md`, Open questions. Confirmation from Tang is pending.

Full log of what was tried, how it did, and what changed:
`EXPERIMENTS.md`

## Run it (Mac, CPU/MPS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pymunk matplotlib numpy torch
python push_t_random.py
python train_world_model.py --episodes 100 --epochs 10
```

## Files

- `push_t_random.py` — Step 0, one random Push-T episode, saves `push_t_random.png`
- `train_world_model.py` — Step 1, collects transitions, fits the MLP world model, saves `world_model.pt`, `world_model_metrics.json`, `world_model_loss.png`
- `train_policy.py` — Step 2, policy trained through the frozen world model (BPTT, H = 32, critic bootstrap), evaluated in the real simulator, saves `policy.pt`, `policy_metrics.json`
- `train_policy_v2.py` — Step 2 v2, shaped reward + near-contact starts, saves `policy_v2.pt`, `policy_metrics_v2.json`
- `diagnose_policy.py` — print what a trained policy actually does in the real simulator
- `heuristic_policy.py` — hand-coded push, straight to behind spot
- `heuristic_v2.py` / `heuristic_v3.py` — waypoint heuristics, see `EXPERIMENTS.md` for why they failed
- `push_from_behind.py` — push straight from behind, no approach, saves `push_from_behind_metrics.json`
- `EXPERIMENTS.md` — experiment log: what was tried, how it did, what changed
- `push_t_explained.png` / `push_t_before_after.png` — diagrams of the problem, start vs success vs failure
