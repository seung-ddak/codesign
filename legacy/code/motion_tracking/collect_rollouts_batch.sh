#!/usr/bin/env bash
# Collect the agreed three-design dataset and publish each validated batch.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TMPDIR="$PWD/.runtime/tmp" MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" runs/pipeline_v1/rollouts
python=../transformer-transformer/.venv/bin/python
for design in e000 e001 e002; do
  for mode in primitive mixed; do
    # e000 mixed seed 0 is already collected and must be retained.
    [[ "$design:$mode" == e000:mixed ]] && continue
    prefix=primitive
    episodes=304
    if [[ "$mode" == mixed ]]; then prefix=seq; episodes=256; fi
    output="runs/pipeline_v1/rollouts/$design/${prefix}_seed0.npz"
    log="runs/pipeline_v1/rollouts/${design}_${mode}_seed0.log"
    [[ ! -e "$output" && ! -e "$log" ]] || { echo "Existing batch: $output" >&2; exit 1; }
    "$python" code/motion_tracking/collect_rollouts.py runs/pipeline_v1/experts/*__r00 \
      --design "$design" --mode "$mode" --episodes "$episodes" --seed 0 > "$log" 2>&1
    "$python" code/motion_tracking/check_rollouts.py "$output" >> "$log" 2>&1
    git diff --cached --quiet || { echo 'Unrelated staged changes; refusing batch commit' >&2; exit 1; }
    git add -- "$output" "${output%.npz}_pairs.csv" "$log"
    git commit -q -m "Add validated $design $mode rollouts (seed 0)"
    git push origin HEAD:main
    echo "Published $design $mode $(git rev-parse --short HEAD)"
    tail -2 "$log"
  done
done
