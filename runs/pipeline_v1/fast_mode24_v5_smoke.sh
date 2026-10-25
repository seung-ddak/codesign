#!/bin/bash
# v5 orig / paired TT 학습 종료 대기 후 같은 조건 (v4 e_rest_s0_try1 과 동일: home S0, zigzag 2s, 후보 4, scale 0 1) smoke.
set -u
cd /mnt/gstore/home/xiangyue/RL/codesign
while tmux has-session -t "=tt_v5_orig" 2>/dev/null || tmux has-session -t "=tt_v5_paired" 2>/dev/null; do sleep 30; done
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
P=../transformer-transformer/.venv/bin/python
for v in orig paired; do
  R=runs/pipeline_v1/fast_mode24_v5_$v
  [ -f $R/tt_joint/checkpoint.pt ] || { echo "MISSING $R/tt_joint/checkpoint.pt"; exit 1; }
  O=$R/smoke_zigzag_joint
  $P code/tt_dynamics/generate_task.py --ckpt $R/tt_joint --out $O --expert-config runs/pipeline_v1/experts_wide16/walkForwardSlow__r00 \
    --anchor-s0 --joint-full-guidance --no-s0-align --seconds 2 --candidates 4 --task zigzag --scales 0 1 --e-rest-s0 || exit 1
  $P code/tt_dynamics/check_fast_smoke.py --out $O --guided-scale 1 --seconds 2
done
