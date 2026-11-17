#!/bin/bash
# TT 학습 창 8 step (2026-11-17 사용자 결정). 나머지 인자는 s9_tt_m4 와 같음.
cd /home/cgr-ugrad-2026/quad_ws/codesign
/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python -u code/tt_dynamics/train_tt.py --data chain/runs/s8_collect/G_*/*.npz \
  --out check/tt_h8/tt --horizon 8 --steps 200000 --batch 32 --norm minmax --clip 1 --sampler ddim --modes 0 1 2 4
echo END
