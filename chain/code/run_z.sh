#!/bin/bash
# 2026-11-17 z 기준 체인: s4 (z 분류 + 부족분 수집) → s5z (assignment) → run_100hz.sh (s5h 리타겟 + s6h 학습, 450M) → s7h 평가 (--wait).
# 사용: tmux new-session -d -s zchain "bash chain/code/run_z.sh"   로그: chain/logs/zchain.log, chain/logs/chain.log
set -e
cd "$(dirname "$0")"
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
$PY -u s4_calm.py --max-rounds 12
$PY -u s5z_assign.py
bash run_100hz.sh
$PY -u s7h_eval100.py --wait > ../logs/s7h.log 2>&1; echo "EXIT $?" >> ../logs/s7h.log
