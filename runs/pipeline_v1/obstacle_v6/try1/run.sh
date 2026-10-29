#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/../../.."
R=runs/pipeline_v1/obstacle_v6/try1
if [ -f "$R/summary.json" ]; then
  echo 'Existing result preserved; use a new output directory for another experiment.' >&2
  exit 1
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl JAX_PLATFORMS=cpu
export MPLCONFIGDIR=/tmp/obstacle_mpl XLA_PYTHON_CLIENT_PREALLOCATE=false
P=../transformer-transformer/.venv/bin/python
$P code/tt_dynamics/generate_task.py --device cpu \
  --ckpt runs/pipeline_v1/fast_mode24_v6_steps60k/tt_joint --out "$R" \
  --expert-config runs/pipeline_v1/experts_wide16/walkForwardSlow__r00 \
  --anchor-s0 --joint-full-guidance --no-s0-align --seconds 4 --candidates 2 \
  --task obstacle --scales 0 10 --e-rest-s0
$P "$R/render_scene.py"
$P "$R/build_report.py"
