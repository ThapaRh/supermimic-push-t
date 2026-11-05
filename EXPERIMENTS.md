# supermimic-push-t — experiment log

This log is the record of what was tried, how it did, and what changed
next. It is written as it happens, including the failures.

## Method, in brief

SuperMimic (Liao et al., arXiv 2610.28480, Oct 2026) learns one
differentiable world model and trains the policy by backpropagation
through short imagined rollouts, with a critic bootstrapping the
return after the imagined horizon.

Transfer to Push-T:
- State (15): pusher x,y and velocity, T-block x,y, cos/sin angle and
  velocity, target x,y and cos/sin angle. All scaled to roughly [-1, 1].
- Action (2): pusher velocity, scaled by 150 px/s.
- Reward: -(position error / 512 + 0.5 * angle error). 0 is perfect.
- Success: position error < 30 px and angle error < 0.3 rad.
- World model: MLP, hidden 128, trained on k = 8-step windows (paper Eq. 1).
- Policy training: imagined horizon H = 32, critic with lambda = 0.95,
  gamma = 0.99 (paper Eq. 2/3, SHAC-like).
- Compute: Mac, CPU/MPS, low-dim state, small MLPs.
- Judge: success rate and final error in the REAL pymunk simulator
  over 20 episodes, varied starts. Imagined return is a training
  signal only, never a result.

## Log

### Step 0 — Random pusher baseline
File: `push_t_random.py`.

What was tried: one random Push-T episode, random action every 20 steps,
300 steps max, in pymunk (512x512 table, T from a 120x30 top bar +
30x90 stem, pusher circle r=20, target 256,256, angle 0).

How it did:
- T-block final (174, 144), target (256, 256)
- Position error 139 px, angle error 0.26 rad
- Success: False

What this means: random fails, as expected. This is the baseline
learning has to beat. Output: `push_t_random.png`.

### Step 1 — World model
File: `train_world_model.py`.
Run: `python3 train_world_model.py --episodes 100 --epochs 10`, on MPS.

What was tried: collect 100 random episodes (30,000 transitions),
train 80 / held-out 20. Fit one MLP world model f(s,a) -> (s', r, done)
on 8-step autoregressive windows.

How it did:
- Loss 0.4537 (epoch 1) -> 0.0069 (epoch 10)
- 1-step T-position error, held-out: 15.3 px
- 8-step T-position error, held-out: 17.7 px

What this means: the model is honest and stable over 8 steps, error
grows only slightly with horizon. It is good enough to train a policy
through, with a caveat: 15 px is about half the 30 px success
threshold, and most steps in random data are non-contact, so contact
error is likely higher than this average. Outputs: `world_model.pt`,
`world_model_metrics.json`, `world_model_loss.png`.

### Step 2 v1 — Policy through the frozen world model
File: `train_policy.py`.
Run: `python3 train_policy.py --iters 200`.

What was tried: freeze the Step 1 world model, imagine H = 32 steps
from start states collected in random episodes (50 episodes,
15,000 states), policy lr 3e-4, critic lr 1e-3, no reward shaping.

How it did:
- Imagined return: -11.9 (iter 1) -> -15.7 (iter 200). Worse, not better.
- Critic loss: 21.8 -> 4.4. The critic learned, the policy did not.
- Real simulator, 20 episodes: success 0/20 (0%)
- Real final position error: 152.3 px mean
- Real final angle error: 0.26 rad mean

What this means: v1 did not learn to push. The policy is
essentially random. This is a recorded failure, not a hidden one.

Bug fixed on the way: first run crashed with
`RuntimeError: Trying to backward through the graph a second time`.
Policy and critic losses shared one critic forward pass, and were
back-propagated separately. Fix: add the two losses and do one
backward pass. The bootstrap value in the policy objective is
detached, so the policy loss does not push the critic around.

Diagnosis for the failure: the reward only cares about where the
T-block is. Until the pusher touches the T, the T does not move, so
the reward does not change, so the policy gets no gradient telling
it to go TO the T first. From a far start, 32 imagined steps is not
enough to stumble into contact by chance.

Outputs: `policy.pt`, `policy_metrics.json`.

### Step 2 v2 — Policy with shaped reward + near-contact starts
File: `train_policy_v2.py`.
Run: `python3 train_policy_v2.py --iters 300`.

What changed, and why:
1. Shaped training reward: model reward - 0.3 * (pusher-to-T distance).
   Gives a gradient to approach the T first. Evaluation is still only
   on the original success metric, so the shaping is a training aid,
   not a fudged score.
2. Near-contact starts: 80% of imagined rollouts start with the
   pusher placed 60-120 px from the T, so contact happens inside
   H = 32 steps. 20% are far random starts, so approach is not
   forgotten.
3. Policy learning rate lowered to 1e-4 (was 3e-4), since v1 drifted worse.
4. New logging: imagined T error at start vs end of a rollout, and
   pusher-to-T distance at the end, so we can see whether the policy
   is actually moving the T in imagination.

How it did:
- Imagined return: -17.2 (iter 1) -> -22.2 (iter 300). Worse, not
  better. Note: shaped reward makes this number not directly
  comparable to v1's -11.9 -> -15.7.
- Critic loss: 22.6 (iter 100) -> 7.7 (iter 300).
- Imagined T error: ~150 px at start -> ~159-162 px at end of a
  32-step imagined rollout. The T moved away from target, not toward it.
- Imagined pusher-to-T distance at end: ~315 px. The pusher did not
  approach the T in imagination, despite 80% near-contact starts.
- Real simulator, 20 episodes: success 0/20 (0%)
- Real final position error: 152.3 px mean
- Real final angle error: 0.26 rad mean

What this means: the real-simulator numbers are identical to v1
to all decimal places (152.267 px, 0.258 rad). That means the
T-block never moved in the real evaluation in either version.
The initial T position alone gives ~150 px error, so both policies
are effectively inert in the real simulator — the pusher never
reaches / never moves the T. v2 did not fix the underlying problem;
shaping and near starts were not enough, because the world model's
imagined dynamics do not transfer to real contact, and/or the
policy gradient through the model is pointing the wrong way.

Outputs: `policy_v2.pt`, `policy_metrics_v2.json`.

### Diagnosis — `diagnose_policy.py --policy policy_v2.pt`
3 real episodes, seeds 1000-1002.

What it showed, episode by episode:
- Ep 0: T (169, 147), pusher (366, 385). Action starts (-0.27, -0.18),
  then (-0.76, +0.42), (-0.75, +0.40). Pusher goes to (18, 464),
  top-left corner. T never moves, T-target stays 139 px throughout.
- Ep 1: T (170, 179), pusher (372, 365). Action ~(-0.1, -0.3) throughout.
  Pusher goes straight down to (347, 181), x never moves toward the T.
  Pusher-T distance 274 -> 177 px, never contacts. T-target stays 115 px.
- Ep 2: same pattern as Ep 0, pusher to (20, 479), top-left corner.
  T-target stays 142 px throughout.

What this means: the policy is NOT inert. Actions are large
(-0.7, +0.4 etc) and the pusher moves hundreds of pixels. It moves
in the WRONG direction, to a corner, away from the T, instead of
toward it. To reach the T from (366, 385), the action should be
roughly (-0.6, -0.8). The policy gives (-0.7, +0.4) instead —
left is right, up/down is flipped. The T never gets touched, so
its error never changes, which explains the identical v1/v2 metrics.

### Heuristic — `heuristic_policy.py`
20 episodes, seeds 1000-1019.

How it did:
- Success 0/20 (0%)
- Real final position error: 286.1 px mean (284-288 px every seed)
- Real final angle error: 0.24 rad mean (0.00 rad on 12 of 20 seeds)

What this means: the heuristic also failed, and it failed WORSE
than random (random / learned: ~152 px). It pushed the T AWAY
from the target, to about 287 px away, consistently.

Why: the heuristic goes straight from the pusher start (380, 380)
to the "behind the T" spot (T - 40 px toward the target).
That straight line passes THROUGH the T at (150, 150). So the
pusher hits the T from the northeast side first, on the way to
the behind spot, and pushes it southwest, away from the target.
The T then keeps getting pushed southwest as the behind spot
moves with it, until it ends up near the southwest corner.
Angle error 0.00 on many seeds is consistent with the T ending
up against a wall / corner, aligned.

This is informative: approach navigation matters. To push the T
northeast, the pusher must first go AROUND the T to the southwest
side, not straight through it. Neither the learned policy nor
this naive heuristic does that.

### Push from behind — `push_from_behind.py`
20 episodes, seeds 1000-1019.
Pusher placed 50 px behind the T, push straight toward target, no navigation.

How it did:
- Success 12/20 (60%)
- Real final position error: 112.7 px mean
- Real final angle error: 0.89 rad mean

Pattern, seed by seed:
- Successes (12): pos err 28-30 px, angle 0.00-0.29 rad,
  T final ~(237, 235). The episode stops as soon as it enters
  the success zone (pos < 30, angle < 0.3), so it stops just
  short of the target centre. That is why successes cluster at 28-30 px.
- Failures (8): pos err 207-286 px, angle 1.07-3.10 rad,
  T final ~(400-465, 400-451), northeast corner area. The T
  spun and veered off, overshot the target, and ended in a corner.

What this means: pushing physics DOES work. 60% success with
no learning at all, just a straight push from behind. So the
env is solvable, and approach is the hard part. The failures
are angle / stability failures: when the initial T angle is
larger, the pusher contacts off-centre, the T rotates, veers,
and overshoots. A policy that can get behind the T, and then
correct angle while pushing, should beat 60%.

### Heuristic v2 — waypoint around south side
File: `heuristic_v2.py`. 20 episodes, seeds 1000-1019.

How it did:
- Success 0/20 (0%)
- Real final position error: 152.3 px mean
- Real final angle error: 0.26 rad mean
- Per-seed pos errors: 139, 115, 142, 146, 170, 136, 175, 157,
  149, 185, 136, 165, 171, 131, 171, 157, 144, 132, 146, 177 px

What this means: T never moved again. The per-seed errors are
exactly the initial errors, and the mean is identical to v1/v2
learned policies (152.267 px, 0.258 rad).

Bug: the v2 stage logic was stateless and oscillated. At the waypoint
(289, 27) the pusher was still 140 px from the T, so the rule
"if far from waypoint AND far from T, go to waypoint" kept
re-triggering. The pusher flipped between going to the waypoint
and going to the behind spot, jittering at (263, 44) forever,
never reaching the T.

### Heuristic v3 — waypoint with a latch
File: `heuristic_v3.py`. 20 episodes, seeds 1000-1019.

How it did:
- Success 0/20 (0%)
- Real final position error: 236.9 px mean
- Real final angle error: 1.14 rad mean

What this means: the latch fixed the oscillation, but the pusher
now arrives at the behind spot at full speed, from the southeast,
moving northwest. It hits the T sideways, pushes it northwest
away from the target, and chases it as it rotates. Trace, seed 1000:
pusher (202, 91), T (168, 148), then T moves to
(51, 367), angle -1.48 rad. Stage 1 never transitions
to stage 2, because the pusher never gets within 40 px of the T
without pushing it away first.

Lesson: approach must be slow and aligned with the push direction,
not just "get to the behind spot at full speed". Contact geometry
is the hard part, for heuristics and for learned policies alike.
