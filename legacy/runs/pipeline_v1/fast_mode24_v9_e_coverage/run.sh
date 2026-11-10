#!/bin/bash
# v9: v8 A 와 같은 구조(--film)·60k. 바뀐 것 = 학습 형상 16 + 새 32 (Sobol 13좌표). 시험 형상 8 은 학습 제외.
# 사전 기준 (시험 형상, A 고정, 시작 30, v8 A 대비): 움직임 일치도 height·tilt·linvel +0.1 이상, 틀린 E 벌점 증가, 기존 게이트 통과.
set -eu
cd "$(dirname "$0")/../../.."
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=code/tt_dynamics
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/v9_mpl
P=../transformer-transformer/.venv/bin/python
R=runs/pipeline_v1/fast_mode24_v9_e_coverage
V8=runs/pipeline_v1/fast_mode24_v8_film
names() { python3 -c "print(' '.join(f'n{i:02d}' for i in range($1,$2)))"; }
$P code/tt_dynamics/collect_e_swap.py --out $R/data_train --designs $R/designs_new.json --targets $(names 0 32) --episodes 8
$P code/tt_dynamics/collect_e_swap.py --out $R/data_test --designs $R/designs_new.json --targets $(names 32 40) --episodes 32
DATA="runs/pipeline_v1/wide16_action/rollouts/e*/primitive_seed0.npz runs/pipeline_v1/e_swap/full/e*/primitive_seed0.npz $R/data_train/n*/primitive_seed0.npz"
[ -f $R/tt_joint/checkpoint.pt ] || $P code/tt_dynamics/train_tt.py --data $DATA --out $R/tt_joint --horizon 32 --steps 60000 --batch 32 --eval-episodes 8 --modes 2 4 --diff-steps 5 --norm minmax --clip 1 --sampler ddim --film
$P $R/measure_given_a.py $R/given_a_test.json "{\"v9\": [\"$R/tt_joint\", 1]}" $R/data_test
$P $R/measure_given_a.py $R/given_a_train_shapes.json "{\"v9\": [\"$R/tt_joint\", 1]}"
$P code/tt_dynamics/e_sensitivity.py --ckpt $R/tt_joint --out $R/sensitivity_v9.json
D=$(python3 -c "import json;print(' '.join(k for k in json.load(open('runs/pipeline_v1/retargeting/designs.json')) if k.startswith('e') and k!='e003'))")
O=$R/check_w1; GEN="code/tt_dynamics/generate_task.py --ckpt $R/tt_joint --expert-config runs/pipeline_v1/experts_wide16/walkForwardSlow__r00 --anchor-s0 --joint-full-guidance --e-rest-s0 --no-s0-align"
$P $GEN --no-render --out $O/smoke_zigzag_joint_c16 --seconds 2 --candidates 16 --task zigzag --scales 0
for task in zigzag circle; do
  $P $GEN --no-render --out $O/fixedE_${task}_4s --seconds 4 --scales 10 --task $task --candidates 4 --fixed-e $D
  $P $GEN --out $O/joint_${task}_4s --seconds 4 --scales 10 --task $task --candidates 16
done
cp $V8/A/check_w1/analyze.py $O/; $P $O/analyze.py
echo V9_ALL_OK
