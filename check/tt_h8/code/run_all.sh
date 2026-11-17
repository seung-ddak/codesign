#!/bin/bash
# 학습 (train.sh) 끝나면: 과제 3 개 × {h8 (매 7 step 다시 생성), h64 (같은 7 step, 비교)} open-loop vs MPC. 후보 8, 유도 0·1.
cd /home/cgr-ugrad-2026/quad_ws/codesign/check/tt_h8; source ../../code/env_osmesa.sh
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
until grep -q '^END' logs/train.log; do sleep 60; done
for task in straight circle zigzag; do
  [ -f gen/h8_e7/$task/summary.json ] || $PY -u code/mpc_generate.py --ckpt tt --task $task --every 7 --out gen/h8_e7/$task
  [ -f gen/h64_e7/$task/summary.json ] || $PY -u code/mpc_generate.py --ckpt ../../chain/runs/s9_tt_m4/tt --task $task --every 7 --out gen/h64_e7/$task
done
echo GEN_END
