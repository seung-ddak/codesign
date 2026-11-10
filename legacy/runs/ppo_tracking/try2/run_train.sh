# PPO 트래킹 try2 학습. 멈춘 뒤 같은 명령으로 다시 실행하면 learner.msgpack 에서 이어 간다.
#   tmux new-session -d -s ppo_try2_train "bash runs/ppo_tracking/try2/run_train.sh"
set -e
cd /home/cgr-ugrad-2026/quad_ws/codesign/code/motion_tracking
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
O=/home/cgr-ugrad-2026/quad_ws/codesign/runs/ppo_tracking/try2
test $(ls $O/ready | wc -l) -eq 20
VISIT=$(ls -d $O/stages/[0-9]* 2>/dev/null | wc -l)   # 이전 실행 횟수 = 이번 회차 번호
STEP=$(python3 -c "import json;print(json.load(open('$O/learner.json'))['step'])" 2>/dev/null || echo 0)
echo "START train visit $VISIT from step $STEP $(date '+%F %T')" >> $O/chain.log
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -u shared_calm_train.py --out $O --stage 0 --visit $VISIT \
  --steps 100000000000 --seconds 1e12 >> $O/train.log 2>&1
echo "END train visit $VISIT $(date '+%F %T')" >> $O/chain.log
