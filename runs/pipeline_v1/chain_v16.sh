#!/bin/bash
# remind 8-49 Δ-loss A/B: (0) A 쌍 수집 종료 대기 → (1) prepare_v16 → (2) twin 학습·평가 → (3) delta 학습·평가.
# 한 단계가 실패하면 이후 의존 단계는 건너뛴다 (delta 평가는 twin 평가 결과를 기준으로 쓰므로 순서 고정).
cd /mnt/gstore/home/xiangyue/RL/codesign_cleanup || exit 9
PY=../transformer-transformer/.venv/bin/python
EVAL=runs/pipeline_v1/eval_v12parity/run_eval.py
while tmux has-session -t "=v16_collect_pairs" 2>/dev/null; do sleep 30; done
n=$(ls runs/pipeline_v1/action_pairs_wide16/*_seed0.npz 2>/dev/null | wc -l)
[ "$n" -eq 16 ] || { echo "action_pairs_wide16 has $n/16 files"; exit 2; }
set -e
echo "=== $(date '+%F %T') prepare";    $PY runs/pipeline_v1/prepare_v16.py
echo "=== $(date '+%F %T') twin_train"; $PY -u runs/pipeline_v1/fast_mode24_v16_twin/run_train.py
echo "=== $(date '+%F %T') twin_eval";  $PY -u $EVAL runs/pipeline_v1/fast_mode24_v16_twin
echo "=== $(date '+%F %T') delta_train"; $PY -u runs/pipeline_v1/fast_mode24_v16_delta/run_train.py
echo "=== $(date '+%F %T') delta_eval"; $PY -u $EVAL runs/pipeline_v1/fast_mode24_v16_delta
echo "=== $(date '+%F %T') CHAIN DONE"
