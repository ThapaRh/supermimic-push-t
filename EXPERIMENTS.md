# supermimic-push-t — experiment log

Project: apply the SuperMimic framework to standard Push-T.
Test for: Prof. Chengzhou Tang, SPINS Lab, University of Manitoba.
Deadline: Mon Nov 16, 2026.
Repo: https://github.com/ThapaRh/supermimic-push-t

This log is the record of what was tried, how it did, and what changed
next. It is written as it happens, including the failures. The final
report to Tang will be built from this log, not from memory.

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
- Compute: Mac only, CPU/MPS, low-dim state, small MLPs.
- Judge: success rate and final error in the REAL pymunk simulator
  over 20 episodes, varied starts. Imagined return is a training
  signal only, never a result.

## Log

### Step 0 — Random pusher baseline
Date: Nov 5, 2026, ~11:18 AM. File: `push_t_random.py`.

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
Date: Nov 5, 2026, ~11:21 AM. File: `train_world_model.py`.
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

Note: a `UserWarning` about converting a tensor with requires_grad
to a scalar appeared in the first run. It was cosmetic, fixed by
using `.detach()` in the loss logging.

### Step 2 v1 — Policy through the frozen world model
Date: Nov 5, 2026, ~11:50 AM. File: `train_policy.py`.
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
Date: Nov 5, 2026. File: `train_policy_v2.py`. Status: running / to run.
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

Hypothesis: with contact inside the horizon and a gradient to
approach, imagined return should rise (less negative) and imagined
T error should fall start -> end, and real success should rise above 0%.

How it did: _to fill in when the run finishes — imagined return
first -> final, critic loss, real success / 20, real final position
and angle error._

Outputs: `policy_v2.pt`, `policy_metrics_v2.json`.

## Open questions for Tang (held, not yet sent)

- Are observations state or vision?
- What is the deliverable, and what evaluation will he run?
- Is there lab compute beyond a Mac, or plan CPU-only?
- Can he share the link to the zita-ch implementation he mentioned?

Cadence question sent Nov 3 in lab Slack, awaiting reply. Do not
send the above as a batch unless a real blocker arises.

## Next steps

1. Finish v2, record its real success rate honestly, even if it is 0%.
2. If v2 still fails, next diagnostics: check world-model error
   specifically on contact steps vs non-contact steps, and check
   whether the policy moves the pusher toward the T at all.
3. Push this log, the new policy file, and updated README to GitHub.
4. Build the short final report from this log before Nov 16.
