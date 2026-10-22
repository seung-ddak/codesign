#!/usr/bin/env bash
# Minimal experiment only: one fresh 2s TT->MJX physical interface check.
# This MUST NOT report a corrected initial state as raw TT feasibility.
set -euo pipefail
cd "$(dirname "$0")/../.."
python=${QUICK_PYTHON:-../transformer-transformer/.venv/bin/python}
ckpt=${QUICK_CKPT:-runs/pipeline_v1/fast_mode24_v4/tt_joint}
root=${QUICK_ROOT:-runs/pipeline_v1/fast_mode24_v4_geometry_gate_try1}
task=${QUICK_TASK:-zigzag}
out="$root/smoke_${task}_joint"
if [[ ! -f "$ckpt/checkpoint.pt" ]]; then
  echo "STOP: checkpoint missing: $ckpt/checkpoint.pt; no TT retraining allowed in this experiment" >&2
  exit 2
fi
if [[ -e "$out/summary.json" ]]; then
  echo "STOP: existing results cannot be overwritten: $out. Set a new QUICK_ROOT." >&2
  exit 2
fi
echo "1/2: verify CPU geometry and raw-vs-assisted report contracts"
"$python" -m pytest -q \
  tests/test_tt_physics_semantics.py \
  tests/test_fast_smoke_audit.py \
  tests/test_joint_generation_gate.py \
  tests/test_tt_rollout_loader.py \
  tests/test_initial_geometry_gate.py \
  tests/test_shared_e_guidance.py \
  tests/test_s0_geometry.py \
  tests/test_s0_height_clamp.py || {
  echo "STOP: CPU regression tests failed; no GPU simulation launched" >&2
  exit 1
}
echo "2/2: same trained TT, opt-in bounded MuJoCo geometry, airborne gate, 2s MJX"
export QUICK_CKPT="$ckpt" QUICK_ROOT="$root" QUICK_SMOKE_SECONDS=2
export QUICK_CANDIDATES=${QUICK_CANDIDATES:-4}
export QUICK_GEOMETRY_S0=1 QUICK_ALLOW_AIRBORNE_S0=1
export QUICK_MAX_GEOMETRY_LIFT_M=${QUICK_MAX_GEOMETRY_LIFT_M:-0.04}
set +e
bash code/tt_dynamics/run_quick_validation.sh smoke
code=$?
set -e
if [[ -f "$out/smoke_check.json" ]]; then
  "$python" - "$out/smoke_check.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("2s MJX completed:", d["guided_completed_MJX_count"])
print("Unmodified TT->MJX completions:", d["guided_completed_unmodified_MJX_count"])
print("Projection-assisted completions:", d["guided_completed_projection_assisted_MJX_count"])
print("Strict raw TT->MJX verified:", d["strict_raw_TT_to_MJX_verified"])
print("Airborne initial candidates:", d["guided_airborne_start_count"])
PY
fi
if (( code != 0 )); then
  echo "STOP: 2s validation failed. See $out/summary.json and smoke_check.json; no 10s run." >&2
  exit 1
fi
echo "2s MJX feasibility gate passed. Inspect strict_raw_TT_to_MJX_verified before interpreting results."
echo "Results: $out/summary.json and $out/smoke_check.json"
