# supermimic-push-t

The SuperMimic training loop (Liao et al., arXiv 2610.28480), applied to the Push-T task.

SuperMimic learns a differentiable world model and trains the policy by backpropagating through short imagined rollouts, with a critic bootstrapping the tail. Here the same loop is applied to Push-T, a 2D task where a round pusher pushes a T-shaped block to a target pose.

## Start here: `supermimic_v3_update/`

That folder has the current, working version. `V1_V2/` keeps the earlier attempts that did not learn, along with their diagnostics.

## Results (real simulator, simplified Push-T success rule)

| Setting | Held-out success (seeds 2000-2099) |
|---|---|
| v3.1, H = 16, seed 0 | **99/100** |
| v3.1, H = 16, random T pose and angle (seeds 3000-3099) | **92/100** |
| H = 16, 3 training seeds | 99%, 99%, 100% (mean 99.3%) |
| H = 16, no critic | 54% |
| H = 8 | 90% (single seed) |
| H = 16, near-sparse reward | 81% (single seed, learns slower) |
| Push straight from behind (baseline) | 15% |
| v1 / v2 | 0% |

Main findings: the critic is essential (about 50% without it, in two separate runs), H = 16 is reliable across training seeds, and a near-sparse reward learns more slowly. Final-success differences for H = 8 and near-sparse are within run-to-run noise: an earlier unsaved run gave 100% for both. Those ablations use one training seed each; see `EXPERIMENTS.md`, Step 5.

## Reproduce

```bash
cd supermimic_v3_update
python3 -m venv .venv && source .venv/bin/activate
pip install pymunk matplotlib numpy torch
python3 supermimic_v3.py --iters 4000 --horizon 16 --tag v31_h16      # ~20 min on 2 CPU cores
python3 eval_v3.py --ckpt policy_v31_h16.pt --episodes 100 --seed0 2000
```

To rerun all Step 5 ablations (about 20 min with two runs in parallel): `bash run_ablations.sh`

## Layout

- `supermimic_v3_update/`: current version
  - `pusht_env.py`: Push-T environment (real T shape, 10 Hz control)
  - `supermimic_v3.py`: online loop (collect, fit world model, BPTT policy and critic)
  - `eval_v3.py`: evaluation in the real simulator
  - `EXPERIMENTS.md`: full log of what was tried, how it did and what changed (Steps 0-5)
- `V1_V2/`: earlier attempts (v1, v2, heuristics, diagnostics)

## Caveats

- The success rule is simplified (position error < 30 px and angle error < 0.3 rad, and the episode ends on entering that zone). The standard gym-pusht 95% coverage metric has not been tested yet.
- A pure 0/1 sparse reward has not been tested yet.
