#!/bin/bash
# A 쌍 (같은 s0·E, 다른 A: orig/swap/offset) 을 wide16 16 형상 전부에서 수집 (remind 8-49, Δ-loss 데이터).
# 기존 action_pairs/ (e000~e002, 옛 rollouts/seq 소스) 와 섞지 않고 새 폴더에 통일된 소스 (wide16_action primitive) 로 모은다.
set -u
cd /mnt/gstore/home/xiangyue/RL/codesign_cleanup || exit 9
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=.40
PY=../transformer-transformer/.venv/bin/python
OUT=runs/pipeline_v1/action_pairs_wide16
mkdir -p "$OUT"
fail=0
for d in e000 e001 e002 e004 e005 e006 e007 e008 e009 e010 e011 e012 e013 e014 e015 e016; do
  f="$OUT/${d}_seed0.npz"
  [ -f "$f" ] && { echo "skip $d (exists)"; continue; }
  echo "=== $(date '+%T') $d"
  $PY code/tt_dynamics/collect_action_pairs.py --source runs/pipeline_v1/wide16_action/rollouts/$d/primitive_seed0.npz \
     --design $d --expert-config runs/pipeline_v1/experts_wide16/walkForwardSlow__r00 --starts 192 --horizon 32 --seed 0 --out "$f" \
     || { echo "FAIL $d"; fail=$((fail+1)); }
done
echo "=== DONE failures=$fail"; exit $fail
