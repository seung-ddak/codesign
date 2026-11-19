#!/bin/bash
# check (2026-11-19): goal_mpc.py 배관 확인. 후보 2, 짧은 최대 시간, circle 목표 0.05 바퀴 (도달 판정 확인).
cd /home/cgr-ugrad-2026/quad_ws/codesign/check/goal_mpc; source ../../code/env_osmesa.sh
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python; CK=../../chain/runs/s9z_tt_h32/tt
$PY -u code/goal_mpc.py --ckpt $CK --task circle3 --candidates 2 --max-seconds 4 --circle-laps 0.05 --out runs/circle3 &&
$PY -u code/goal_mpc.py --ckpt $CK --task zigzag3 --candidates 2 --max-seconds 3 --out runs/zigzag3 &&
$PY -u code/goal_mpc.py --ckpt $CK --task straight --candidates 2 --max-seconds 3 --out runs/straight
echo "EXIT $?"
