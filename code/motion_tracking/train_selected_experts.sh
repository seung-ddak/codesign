#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
selection=runs/pipeline_v1/experts/selection.csv
root=runs/pipeline_v1/experts
python=../transformer-transformer/.venv/bin/python
export MUJOCO_GL=egl WANDB_MODE=offline XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1

motions=()
while IFS=, read -r motion floor _; do
  [[ "$motion" =~ ^[[:alnum:]_]+$ && "$floor" == 0.0 ]] || { echo "Invalid selection: $motion ($floor)" >&2; exit 1; }
  for design in e000 e001 e002; do
    [[ -f "runs/pipeline_v1/retargeting/$design/00000011/$motion.npz" ]] || { echo "Missing $design/$motion" >&2; exit 1; }
  done
  motions+=("$motion")
done < <(tail -n +2 "$selection")
[[ ${#motions[@]} -gt 0 ]] || { echo "Empty selection" >&2; exit 1; }

for motion in "${motions[@]}"; do
  run="$root/$motion"
  if grep -q "^Completed $motion:" "$root/queue.log"; then
    echo "Skipping completed $motion"
    continue
  fi
  overrides=()
  log="$run/train.log"
  if [[ -d "$run" ]]; then
    ckpt=$(find "$run/wandb" -path '*/files/checkpoints/*/_CHECKPOINT_METADATA' -printf '%h\n' | sort -t/ -k9,9nr | head -n 1)
    [[ -n "$ckpt" ]] || { echo "No checkpoint to resume: $motion" >&2; exit 1; }
    step=${ckpt##*/}
    remaining=$((5242880 - step))
    [[ $remaining -gt 0 ]] || { echo "No steps remaining: $motion" >&2; exit 1; }
    overrides+=("restore_ckpt=$PWD/$ckpt" "train.num_timesteps=$remaining" "hydra.run.dir=$run/resume")
    log="$run/resume.log"
    echo "Resuming $motion from $step ($remaining steps remaining)"
  else
    mkdir -p "$run"
    overrides+=("train.num_timesteps=5242880" "hydra.run.dir=$run")
  fi
  echo "Training $motion"
  WANDB_DIR="$PWD/$run" "$python" code/motion_tracking/train_rl_procedural.py \
    "env.clips=[$motion]" "eval_env.clips=[$motion]" \
    train.num_envs=32 train.batch_size=32 \
    train.num_eval_envs=4 train.num_evals=10 render_final=false "${overrides[@]}" >"$log" 2>&1
  latest=$(find "$run/wandb" -maxdepth 1 -type d -name 'offline-run-*' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
  [[ -n "$latest" && -d "$latest/files/checkpoints" ]] || { echo "Missing checkpoint: $motion" >&2; exit 1; }
  bash code/motion_tracking/eval_multi_designs.sh "$latest" >"$run/eval.log" 2>&1
  echo "Completed $motion: $latest/motion_eval/summary.csv"
done
