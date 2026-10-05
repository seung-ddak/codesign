#!/bin/bash
# TT ctrl 정책이 계단을 오르는지 확인한다.
#   bash scripts/eval_ctrl_stairs.sh <궤적 pkl> [명령속도 인덱스들]
set -u
TT="${CODESIGN_TT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/transformer-transformer}"
PKL="${1:-data/stair_walk_r05.pkl}"
IDXS="${2:-0 1 2}"
SPEEDS=(0.05 0.10 0.15)
cd "$TT"

for idx in $IDXS; do
  echo "=== ${PKL} / 명령 ${SPEEDS[$idx]} m/s ==="
  CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python scripts/evaluate_ctrl.py \
    'ckpt_path=checkpoints/umi_on_legs_plus_plus/d7lra7j0/040.pt' \
    'env@eval_fn.runner.env=umi_on_legs_plus_plus_stairs' \
    'eval_fn.runner.env.robot_generator.choices=[0,0,0,0,0,0,0,0]' \
    eval_fn.runner.env.robot_generator.num_uniforms=13 \
    "eval_fn.runner.env.pickle_path=${PKL}" \
    'eval_fn.hardware_traj_pairs_generator.list1=[10000000]' \
    "eval_fn.hardware_traj_pairs_generator.list2=[${idx}]" \
    eval_fn.num_processes=1 \
    use_torch_compile=false use_flash_attn=false use_mixed_precision=false \
    2>/dev/null \
    | grep -aoE '(metric/pos_err/mean|metric/reward/sum|done/[a-z_]+/any): [-0-9.]+%?'
done
echo "=== done ==="
