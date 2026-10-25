#!/bin/bash
# v5 orig / paired: 후보 16 smoke (v5 smoke 와 같은 조건, 후보 수만 4 -> 16, 렌더 끔).
set -u
cd /mnt/gstore/home/xiangyue/RL/codesign
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
P=../transformer-transformer/.venv/bin/python
for v in orig paired; do
  R=runs/pipeline_v1/fast_mode24_v5_$v
  O=$R/smoke_zigzag_joint_c16
  $P code/tt_dynamics/generate_task.py --ckpt $R/tt_joint --out $O --expert-config runs/pipeline_v1/experts_wide16/walkForwardSlow__r00 \
    --anchor-s0 --joint-full-guidance --no-s0-align --no-render --seconds 2 --candidates 16 --task zigzag --scales 0 1 --e-rest-s0 || exit 1
  $P code/tt_dynamics/check_fast_smoke.py --out $O --guided-scale 1 --seconds 2
done
