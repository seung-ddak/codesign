#!/bin/bash
# 논문 Table 4 (Quadruped, Single-Trajectory) 의 TT 행 재현.
#
# 기준값 (논문): Pos Err 2.6 cm / Orn Err 7.4 deg / Survival 99.4 % / 최적화 5.0 s
# 이 근처가 나와야 배선이 맞는 것이다 (remind §5.16).
#
# 설정 근거 (docs/visualization.md 표):
#   quadruped / 모든 reward / single  ->  guidance.scale 100.0, eta 1.0,
#                                          clip_samples_in_guidance true (기본값)
# V100 제약 (remind §4.1): flash attn·bf16·torch.compile 전부 끈다.
#
#   bash tt_codesign/code/repro_table4_TT.sh [num_seeds] [optimizer]
#     num_seeds : 후보 샘플 수 n (기본 16)
#     optimizer : guided_diffusion (TT/DSG, 기본) | zeroth_order (기준선)

set -u
TT="${CODESIGN_TT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/transformer-transformer}"
cd "$TT"

N="${1:-16}"
OPT="${2:-guided_diffusion}"
CKPT=checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt   # hardware gen (datasets=clean)
PKL=data/july25th2025-huy-20skills-test.pkl              # held-out 20 skill

for f in choice_to_ckpt_path.json umi_on_legs_plus_plus_robotokens umi_on_legs_plus_plus_options; do
  [ -e "$f" ] || { echo "[중단] $f 없음 — rl_policies.zip 압축 해제 먼저"; exit 1; }
done

EXTRA=""
if [ "$OPT" = "guided_diffusion" ]; then
  EXTRA="eval_fn.hardware_optimizer_fn.guidance.scale=100.0 eta=1.0"
  TAG=guided
else
  TAG=zeroth
fi

echo "=== Table 4 ${TAG} | n=${N} | $(date '+%H:%M:%S') ==="
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python \
  scripts/evaluate_hardware_opt_rl.py \
  "tags=[quadruped,tracking_only,${TAG}]" \
  hardware_optimizer@eval_fn.hardware_optimizer_fn="${OPT}" \
  reward_fns@eval_fn.hardware_optimizer_fn.reward_fns=tracking_only \
  eval_fn.hardware_optimizer_fn.num_seeds="${N}" \
  ${EXTRA} \
  use_flash_attn=false use_mixed_precision=false use_torch_compile=false \
  ckpt_path="${CKPT}" \
  pickle_path="${PKL}"
