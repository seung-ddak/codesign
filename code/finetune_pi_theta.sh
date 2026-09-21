#!/bin/bash
# 생성된 형상 하나에 π_θ 를 맞춘다 — "그 형상에 맞는 최적 정책" (D23).
#
# task = 전진 보행. `flat_walk_long.pkl` 의 EE 목표가 +x 로 0.1 m/s 이동하므로
# **`exp_pose_err` 를 따라가는 것이 곧 전진**이다 (별도 progress 항이 필요 없다).
# 종료 조건은 D22 와 일치시킨다: 지정 링크 접촉 + fellover 0.1.
#
# 기준선 = 출고 expert (그 choice 의 연속 변형 2 만 개를 DR 로 학습한 범용 정책).
# 여기서는 **그 expert 에서 출발해 형상 1 개에만** 이어 학습한다 -> 형상 특화 이득만 남는다.
#
#   bash tt_codesign/code/finetune_pi_theta.sh <zarr> <expert_ckpt> [timesteps]

set -u
TT=/mnt/gstore/home/xiangyue/RL/transformer-transformer
cd "$TT"

ZARR="${1:?형상 zarr 경로}"
EXPERT="${2:?출고 expert 체크포인트}"
STEPS="${3:-40000000}"          # 출고 학습은 4e8. fine-tune 은 1/10 에서 시작해 본다
PKL="${4:-data/flat_walk_long.pkl}"   # task = 전진 보행 (EE 목표가 +x 로 0.1 m/s)

[ -e "$ZARR" ] || { echo "[중단] zarr 없음: $ZARR"; exit 1; }
[ -e "$EXPERT" ] || { echo "[중단] expert 없음: $EXPERT"; exit 1; }

echo "=== π_θ fine-tune | $(basename "$ZARR") | steps ${STEPS} | $(date '+%H:%M:%S') ==="
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline .venv/bin/python \
  scripts/train_rl_procedural.py \
  mj_model_data_path="${ZARR}" \
  env.trajectory_file="${PKL}" \
  eval_env.trajectory_file="${PKL}" \
  +train.restore_checkpoint_path="${EXPERT}" \
  train.num_timesteps="${STEPS}" \
  env.termination_fellover_threshold=0.1 \
  env.terminate_on_bad_contact=true \
  eval_envs.termination_fellover_threshold=0.1 \
  'wandb.tags=[pi_theta,shape_specific,walk]'
