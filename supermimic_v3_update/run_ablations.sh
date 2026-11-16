#!/usr/bin/env bash
# Step 5 ablations (see EXPERIMENTS.md). Each run trains for 4000 iters,
# keeps the best checkpoint on eval seeds 1000-1019, then evaluates it on
# held-out seeds 2000-2099.
# Outputs per run: metrics_<tag>.json, policy_<tag>.pt, eval_<tag>_heldout.{json,png}
# Run from this folder:  bash run_ablations.sh
# Two runs at a time by default; set JOBS=1 on a slow machine.
set -euo pipefail
cd "$(dirname "$0")"
JOBS=${JOBS:-2}

COMMON="--iters 4000 --eval_every 250"
RUNS=(
  "h16_seed1|--horizon 16 --seed 1"
  "h16_seed2|--horizon 16 --seed 2"
  "h16_nocritic|--horizon 16 --no_critic"
  "h16_sparse|--horizon 16 --sparse"
  "h8|--horizon 8"
)

run_one() {
  local tag="${1%%|*}" flags="${1#*|}"
  echo "[$(date +%H:%M)] start $tag"
  python3 supermimic_v3.py $COMMON $flags --tag "$tag" > "train_$tag.txt" 2>&1
  python3 eval_v3.py --ckpt "policy_$tag.pt" --episodes 100 --seed0 2000 \
    --plot "eval_${tag}_heldout.png" > "eval_$tag.txt" 2>&1
  echo "[$(date +%H:%M)] done  $tag: $(tail -2 "eval_$tag.txt" | head -1)"
}
export -f run_one
export COMMON

printf '%s\n' "${RUNS[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {}

echo
echo "Held-out success (seeds 2000-2099):"
for r in "${RUNS[@]}"; do
  tag="${r%%|*}"
  python3 - "$tag" <<'EOF'
import json, sys
tag = sys.argv[1]
rows = json.load(open(f"eval_{tag}_heldout.json"))
m = json.load(open(f"metrics_{tag}.json"))
print(f"{tag:14s} held-out {sum(r['success'] for r in rows)}/{len(rows)}   "
      f"best eval {m['best_success']} at iter {m['best_iter']}")
EOF
done
