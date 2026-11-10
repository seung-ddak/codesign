#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export TMPDIR="$PWD/.runtime/tmp" MPLCONFIGDIR="$PWD/.runtime/mpl" XDG_CACHE_HOME="$PWD/.runtime/cache"
export CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu MUJOCO_GL=egl
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$TMPDIR" "$MPLCONFIGDIR"
PYTHON=../transformer-transformer/.venv/bin/python
OUT=runs/pipeline_v1/regression_rest_features
nice -n 10 "$PYTHON" -u "$OUT/prepare.py" > "$OUT/prepare.log" 2>&1
nice -n 10 "$PYTHON" -u code/tt_dynamics/regression_gate.py --data runs/pipeline_v1/pd_rollouts_kp200 --rest-features "$OUT/rest_features.npz" --out "$OUT" > "$OUT/run.log" 2>&1
nice -n 10 "$PYTHON" -u "$OUT/summarize.py" > "$OUT/summary.log" 2>&1
