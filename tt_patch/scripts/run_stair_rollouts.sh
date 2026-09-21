#!/bin/bash
# TT ctrl 정책으로 계단 rollout 을 여러 명령 속도에서 돌린다.
#   bash scripts/run_stair_rollouts.sh <env> <pkl> [traj 인덱스들]
set -u
cd /mnt/gstore/home/xiangyue/RL/transformer-transformer
ENV="${1:-umi_on_legs_plus_plus_stairs}"
PKL="${2:-data/stair_walk_r05.pkl}"
IDXS="${3:-0 1 2}"

for t in $IDXS; do
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/ctrl_rollout_direct.py \
    --env "$ENV" --pkl "$PKL" --traj "$t" --steps 1000 2>/dev/null | tail -7
done
echo "=== ALLDONE ==="
