#!/bin/bash
# v6: v5_orig 와 같은 데이터·설정, 학습량만 20000 -> 60000 step. 이어서 후보 16 smoke (v5 c16 과 같은 조건).
set -eu
cd /mnt/gstore/home/xiangyue/RL/codesign
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
P=../transformer-transformer/.venv/bin/python
R=runs/pipeline_v1/fast_mode24_v6_steps60k
$P code/tt_dynamics/train_tt.py --data runs/pipeline_v1/wide16_action/rollouts/e*/primitive_seed0.npz --out $R/tt_joint \
  --horizon 32 --steps 60000 --batch 32 --eval-episodes 8 --modes 2 4 --diff-steps 5 --norm minmax --clip 1 --sampler ddim
O=$R/smoke_zigzag_joint_c16
$P code/tt_dynamics/generate_task.py --ckpt $R/tt_joint --out $O --expert-config runs/pipeline_v1/experts_wide16/walkForwardSlow__r00 \
  --anchor-s0 --joint-full-guidance --no-s0-align --no-render --seconds 2 --candidates 16 --task zigzag --scales 0 1 --e-rest-s0
$P code/tt_dynamics/check_fast_smoke.py --out $O --guided-scale 1 --seconds 2
