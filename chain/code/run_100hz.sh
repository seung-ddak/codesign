#!/bin/bash
# 100 Hz 추종 정책 본 학습 시작 (2026-11-17 확정 계획: check/tracking_policy/PLAN_100HZ.md).
# 리타겟 (CPU, tmux s5h) 과 학습 (GPU, tmux s6h) 을 동시에 띄운다. 학습은 모션 세트별로 리타겟이 끝나는 대로 시작.
# 사용: bash chain/code/run_100hz.sh            (정책당 450M, 묶음 4 개 번갈아, 방문당 25M)
#       STEPS=50000000 bash chain/code/run_100hz.sh
# 진행 로그: chain/logs/chain.log ('s5h', 's6h'), tmux 출력: chain/logs/s5h.log, chain/logs/s6h.log
# 학습 중에는 git 명령을 쓰지 않는다 (codesign/CLAUDE.md).
set -e
cd "$(dirname "$0")"
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
STEPS=${STEPS:-450000000}
for s in s5h s6h; do
  if tmux has-session -t "=$s" 2>/dev/null; then echo "tmux 세션 $s 이 이미 있음 — 중단"; exit 1; fi
done
if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; then
  echo "GPU 를 쓰는 프로세스가 있음 — 끝난 뒤 다시 실행:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; exit 1
fi
tmux new-session -d -s s5h "bash -c '$PY -u s5h_retarget100.py --workers 16 > ../logs/s5h.log 2>&1; echo EXIT \$? >> ../logs/s5h.log'"
sleep 30  # assignment.json 이 먼저 생기도록
tmux new-session -d -s s6h "bash -c '$PY -u s6h_train100.py --steps $STEPS --chunks 4 --visit-steps 25000000 > ../logs/s6h.log 2>&1; echo EXIT \$? >> ../logs/s6h.log'"
echo "시작: tmux s5h (리타겟), s6h (학습, 정책당 $STEPS)"
