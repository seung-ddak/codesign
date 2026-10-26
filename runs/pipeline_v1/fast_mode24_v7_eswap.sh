#!/bin/bash
# v7: v6 와 같은 설정 (60000 step) + E 반사실 데이터 (remind 8-32). 단변수 = 학습 데이터에 e_swap 추가.
# 수집 → 학습 → 2 s smoke (scale 0, 경로 차 기준) → co-design 검사 (v6 와 같은 조건) → 분석.
set -eu
cd /mnt/gstore/home/xiangyue/RL/codesign
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
P=../transformer-transformer/.venv/bin/python
R=runs/pipeline_v1/fast_mode24_v7_eswap
X=runs/pipeline_v1/e_swap/full
$P code/tt_dynamics/collect_e_swap.py --out $X --episodes 32 --steps 100
$P code/tt_dynamics/train_tt.py --data runs/pipeline_v1/wide16_action/rollouts/e*/primitive_seed0.npz $X/e*/primitive_seed0.npz \
  --out $R/tt_joint --horizon 32 --steps 60000 --batch 32 --eval-episodes 8 --modes 2 4 --diff-steps 5 --norm minmax --clip 1 --sampler ddim
GEN="code/tt_dynamics/generate_task.py --ckpt $R/tt_joint --expert-config runs/pipeline_v1/experts_wide16/walkForwardSlow__r00 \
  --anchor-s0 --joint-full-guidance --e-rest-s0 --no-s0-align --no-render"
$P $GEN --out $R/smoke_zigzag_joint_c16 --seconds 2 --candidates 16 --task zigzag --scales 0
D=$(python3 -c "import json;print(' '.join(k for k in json.load(open('runs/pipeline_v1/retargeting/designs.json')) if k.startswith('e') and k!='e003'))")
for task in zigzag circle; do
  $P $GEN --out $R/codesign_check/fixedE_${task}_4s --seconds 4 --scales 10 --task $task --candidates 4 --fixed-e $D
  $P $GEN --out $R/codesign_check/joint_${task}_4s --seconds 4 --scales 10 --task $task --candidates 16
done
cp runs/pipeline_v1/fast_mode24_v6_steps60k/codesign_check/analyze.py $R/codesign_check/
$P $R/codesign_check/analyze.py
