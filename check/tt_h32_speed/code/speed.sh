#!/bin/bash
# 본학습 (s9z) 설정 그대로 H 32 · 100 Hz · 전체 s8z · split shape 로 2000 step: step 당 시간 측정 (2026-11-18)
cd /home/cgr-ugrad-2026/quad_ws/codesign/code/tt_dynamics
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
$PY -u train_tt.py --data $(ls ../../chain/runs/s8z_collect/rollouts/*/chunk_*.npz) --out ../../check/tt_h32_speed/tt \
  --episode-steps 2000 --dt 0.01 --data-device cpu --split-by shape \
  --horizon 32 --steps 2000 --batch 32 --norm minmax --clip 1 --sampler ddim --modes 0 1 2 4 --eval-episodes 8 --seed 0 \
  | while IFS= read -r l; do echo "$(date +%s.%N | cut -c1-14) $l"; done
echo "EXIT ${PIPESTATUS[0]}"
