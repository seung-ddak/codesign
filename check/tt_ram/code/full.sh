#!/bin/bash
# 전체 s8z 데이터, CPU 보관, EP 2000, 100 Hz 로 train_tt 200 step: 최고 RSS·step 시간 확인
cd /home/cgr-ugrad-2026/quad_ws/codesign/code/tt_dynamics
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
/usr/bin/time -v $PY -u train_tt.py --data $(ls ../../chain/runs/s8z_collect/rollouts/*/chunk_*.npz) --out ../../check/tt_ram/runs/full_shape \
  --episode-steps 2000 --dt 0.01 --data-device cpu --split-by shape --steps 200 --val-every 100 --checkpoint-every 1000 --eval-episodes 16 --seed 0
echo "EXIT $?"
