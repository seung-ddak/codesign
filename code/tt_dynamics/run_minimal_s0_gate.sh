#!/usr/bin/env bash
# Bounded FAST S0-feasibility gate. Does not retrain TT or change its joint E/S/A samples.
# The old two-sample failure is audited first; then a small candidate batch is
# generated and screened. Never adjusts body height or weakens MuJoCo physics.
set -euo pipefail
cd "$(dirname "$0")/../.."
python=${QUICK_PYTHON:-../transformer-transformer/.venv/bin/python}
ckpt=${QUICK_CKPT:-runs/pipeline_v1/fast_mode24_v4/tt_joint}
old=${S0_PREVIOUS_SMOKE:-runs/pipeline_v1/fast_mode24_v4_smoke_audit/smoke_zigzag_joint}
root=${QUICK_ROOT:-runs/pipeline_v1/fast_mode24_v4_s0_gate}
task=${QUICK_TASK:-zigzag}
out="$root/smoke_${task}_joint"
if [[ -e "$out/summary.json" ]]; then
  echo "STOP: output already exists: $out. Set a new QUICK_ROOT; previous results are never overwritten." >&2
  exit 2
fi
mkdir -p "$root"
echo "[1/3] Inspect existing S0 height vs trained normalization / clamp."
"$python" code/tt_dynamics/audit_s0_height_clamp.py \
  --ckpt "$ckpt" --smoke-dir "$old" --out "$root/previous_s0_clamp.json" || {
    echo "STOP: checkpoint or old Smoke data missing; see $root/previous_s0_clamp.json" >&2
    exit 1
  }

echo "[2/3] Run bounded fresh TT samples with the SAME checkpoint. No height projection/retraining."
# One bounded batch; stop if none of the guided candidates has a physics-valid
# initial state and survives the complete 2s MJX open-loop check.
export QUICK_CKPT="$ckpt" QUICK_ROOT="$root" QUICK_SMOKE_SECONDS=2
export QUICK_CANDIDATES=${QUICK_CANDIDATES:-4}
set +e
bash code/tt_dynamics/run_quick_validation.sh smoke
smoke_status=$?
set -e

echo "[3/3] Inspect NEW sampled S0 heights even when the physical test failed."
if [[ -f "$out/scale_0.npz" && -f "$out/scale_1.npz" ]]; then
  "$python" code/tt_dynamics/audit_s0_height_clamp.py \
    --ckpt "$ckpt" --smoke-dir "$out" --out "$root/new_s0_clamp.json" || exit 1
else
  echo "STOP: generation failed before producing both result NPZ files." >&2
  exit 1
fi

if (( smoke_status != 0 )); then
  echo "STOP: 2s feasibility gate failed (exit $smoke_status). Read:" >&2
  echo "  $out/smoke_check.json" >&2
  echo "  $root/new_s0_clamp.json" >&2
  echo "No 10s run was started. Do not claim MJX reward success." >&2
  exit 1
fi
echo "PASS: bounded 2s TT generation + physics + MJX smoke gate."
echo "Results: $out/smoke_check.json and $root/new_s0_clamp.json"
