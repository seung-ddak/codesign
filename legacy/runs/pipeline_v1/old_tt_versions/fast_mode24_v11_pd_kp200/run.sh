#!/usr/bin/env bash
# v11: PD kp200 16형상 재수집 → 전진 부호 gate → data_audit/sha256 → v10 동일 파이프라인 (학습·평가·보고서). remind 8-36.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
P=../transformer-transformer/.venv/bin/python
V11=runs/pipeline_v1/fast_mode24_v11_pd_kp200
"$P" -u code/motion_tracking/collect_pd_rollouts.py --kp 200 --out runs/pipeline_v1/pd_rollouts_kp200
: > "$V11/drift_table.txt"
for d in e000 e001 e002 e004 e005 e006 e007 e008 e009 e010 e011 e012 e013 e014 e015 e016; do
  echo "## $d" >> "$V11/drift_table.txt"
  "$P" -I code/tt_dynamics/check_pd_drift.py "runs/pipeline_v1/pd_rollouts_kp200/$d/primitive_seed0.npz" \
    --ref "runs/pipeline_v1/retargeting/$d/00000011" >> "$V11/drift_table.txt"
done
if grep -q '부호 불일치: [1-9]' "$V11/drift_table.txt"; then echo "GATE_FAIL: 전진 부호 불일치 형상 있음 (drift_table.txt)"; exit 3; fi
"$P" -I "$V11/make_data_audit.py"
unset XLA_PYTHON_CLIENT_ALLOCATOR
"$P" -u "$V11/run_comparison.py"
