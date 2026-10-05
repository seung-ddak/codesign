#!/bin/bash
# 논문 Table 4 (Quadruped, Single-Trajectory) 의 **RL 행**.
#
# TT 행(`repro_table4_TT.sh`)과 **완전히 같은 프로토콜**이고, guidance 를 만드는 주체만 다르다.
#   TT : guided_diffusion  = scale·∇R̂        (DSG)
#   RL : rl_residual       = π_φ(x_t, k)      (학습된 residual, prior 동결)
#
# 비교 기준 (논문 / 우리 TT 행 재현):
#   Pos Err 2.6 / 3.0 cm · Orn Err 7.4 / 8.0° · Survival 99.4 / 99.4 %
#
#   bash tt_codesign/code/repro_table4_RL.sh [num_seeds] [policy_ckpt]

set -u
TT="${CODESIGN_TT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/transformer-transformer}"
cd "$TT"

N="${1:-16}"
POLICY="${2:-${CODESIGN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/runs/pi_phi_try2/policy.pt}"
CKPT=checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt
PKL=data/july25th2025-huy-20skills-test.pkl

[ -e "$POLICY" ] || { echo "[중단] π_φ 체크포인트 없음: $POLICY"; exit 1; }
for f in choice_to_ckpt_path.json umi_on_legs_plus_plus_robotokens; do
  [ -e "$f" ] || { echo "[중단] $f 없음"; exit 1; }
done

echo "=== Table 4 RL | n=${N} | π_φ=${POLICY} | $(date '+%H:%M:%S') ==="
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python \
  scripts/evaluate_hardware_opt_rl.py \
  "tags=[quadruped,tracking_only,rl_guided]" \
  hardware_optimizer@eval_fn.hardware_optimizer_fn=rl_residual \
  reward_fns@eval_fn.hardware_optimizer_fn.reward_fns=tracking_only \
  eval_fn.hardware_optimizer_fn.num_seeds="${N}" \
  eval_fn.hardware_optimizer_fn.policy_ckpt="${POLICY}" \
  eta=1.0 \
  use_flash_attn=false use_mixed_precision=false use_torch_compile=false \
  ckpt_path="${CKPT}" \
  pickle_path="${PKL}"
