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
- `train_policy.py` — Step 2 (next), policy trained through the frozen world model
