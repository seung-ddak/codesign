#!/bin/bash
# v8: v7 과 같은 데이터·60k. A = E 전 층 주입, B = A + E 가리기 15%. 평가 B 는 E 영향 2 배.
set -eu
cd "$(dirname "$0")/../../.."
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform PYTHONPATH=code/tt_dynamics
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/v8_mpl
P=../transformer-transformer/.venv/bin/python
R=runs/pipeline_v1/fast_mode24_v8_film
V7=runs/pipeline_v1/fast_mode24_v7_eswap/tt_joint
DATA="runs/pipeline_v1/wide16_action/rollouts/e*/primitive_seed0.npz runs/pipeline_v1/e_swap/full/e*/primitive_seed0.npz"
TRAIN="--horizon 32 --steps 60000 --batch 32 --eval-episodes 8 --modes 2 4 --diff-steps 5 --norm minmax --clip 1 --sampler ddim"
train() { [ -f "$R/$1/tt_joint/checkpoint.pt" ] || $P code/tt_dynamics/train_tt.py --data $DATA --out $R/$1/tt_joint $TRAIN "${@:2}"; }
train A --film
train B --film --e-dropout 0.15
$P code/tt_dynamics/e_sensitivity.py --ckpt $V7 --out $R/sensitivity/v7.json
$P code/tt_dynamics/e_sensitivity.py --ckpt $R/A/tt_joint --out $R/sensitivity/A.json
$P code/tt_dynamics/e_sensitivity.py --ckpt $R/B/tt_joint --out $R/sensitivity/B_w1.json
$P code/tt_dynamics/e_sensitivity.py --ckpt $R/B/tt_joint --e-cfg 2 --out $R/sensitivity/B_w2.json
D=$(python3 -c "import json;print(' '.join(k for k in json.load(open('runs/pipeline_v1/retargeting/designs.json')) if k.startswith('e') and k!='e003'))")
check() {  # <variant> <e-cfg>
  local O=$R/$1/check_w$2 GEN="code/tt_dynamics/generate_task.py --ckpt $R/$1/tt_joint --expert-config runs/pipeline_v1/experts_wide16/walkForwardSlow__r00 --anchor-s0 --joint-full-guidance --e-rest-s0 --no-s0-align --e-cfg $2"
  $P $GEN --no-render --out $O/smoke_zigzag_joint_c16 --seconds 2 --candidates 16 --task zigzag --scales 0
  for task in zigzag circle; do
    $P $GEN --no-render --out $O/fixedE_${task}_4s --seconds 4 --scales 10 --task $task --candidates 4 --fixed-e $D
    $P $GEN --out $O/joint_${task}_4s --seconds 4 --scales 10 --task $task --candidates 16
  done
  cp runs/pipeline_v1/fast_mode24_v7_eswap/codesign_check/analyze.py $O/
  $P $O/analyze.py
}
check A 1
check B 2
echo V8_ALL_OK
