#!/bin/bash
# jump smoke 학습 + 영상 (tmux·텔레그램 알림 포함). RL 폴더에서: bash codesign/code/run_jump_smoke.sh <run 이름>
set -e
RUN=${1:?run 이름 필요 (예: smoke_try7)}
cd "$(dirname "$0")/../.."
bash scripts/tmux_run.sh "jump_$RUN" "bash -c 'P=transformer-transformer/.venv/bin/python; MUJOCO_GL=egl \$P codesign/code/train_policy.py --out codesign/runs/expert/jump/$RUN --phase jump --seeds-per-choice 1 --timesteps 153600 --num-envs 64 --checkpoints 2 --learning-rate 3e-5 --max-groups 1 --seed 0 && MUJOCO_GL=egl \$P codesign/code/record_jump_rollout.py --run codesign/runs/expert/jump/$RUN'"
