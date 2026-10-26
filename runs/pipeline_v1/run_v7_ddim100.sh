#!/usr/bin/env bash
# Audited V7 follow-up: preserve the successful v6 settings and change ONLY
# diffusion training/sampling from 5 to 100 levels. Keep original runs immutable.
# Stages: prepare | train | smoke | codesign | full | report | all
set -euo pipefail
cd "$(dirname "$0")/../.."

stage="${1:-all}"
case "$stage" in prepare|train|smoke|codesign|full|report|all) ;; *)
  echo "Usage: bash runs/pipeline_v1/run_v7_ddim100.sh {prepare|train|smoke|codesign|full|report|all}" >&2; exit 2;;
esac

P="${V7_100_PYTHON:-../transformer-transformer/.venv/bin/python}"
ROOT="${V7_100_ROOT:-runs/pipeline_v1/v7_ddim100}"
BASE="runs/pipeline_v1/wide16_action/rollouts"
SWAP="runs/pipeline_v1/e_swap/full"
CKPT="$ROOT/tt_joint"
EXPERT="${V7_100_EXPERT:-runs/pipeline_v1/experts_wide16/walkForwardSlow__r00}"
FIXED_SCALE="${V7_100_FIXED_SCALE:-10}"
SEED="${V7_100_SEED:-0}"
mkdir -p "$ROOT"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
command -v "$P" >/dev/null 2>&1 || [[ -x "$P" ]] || { echo "Missing Python: $P" >&2; exit 1; }

# e003 is the UNSEEN morphology and must NOT enter train/validation data.
designs=(e000 e001 e002 e004 e005 e006 e007 e008 e009 e010 e011 e012 e013 e014 e015 e016)
base_files=()
swap_files=()
for design in "${designs[@]}"; do
  base_files+=("$BASE/$design/primitive_seed0.npz")
  swap_files+=("$SWAP/$design/primitive_seed0.npz")
done

require_base() {
  for path in "${base_files[@]}"; do
    [[ -s "$path" ]] || { echo "Missing original 16-morphology rollout: $path" >&2; exit 1; }
  done
}
require_data() {
  require_base
  for path in "${swap_files[@]}"; do
    [[ -s "$path" ]] || { echo "Missing E-swap rollout: $path (run prepare)" >&2; exit 1; }
  done
  "$P" - "$SEED" "${swap_files[@]}" <<'PYDATA'
import sys, numpy as np
for path in sys.argv[2:]:
    with np.load(path) as data:
        if int(data["seed"]) != int(sys.argv[1]) or data["valid"].shape[1] != 100:
            raise SystemExit(f"E-swap collection seed/length mismatch: {path}; preserve it and use matching collection settings")
PYDATA
}
require_checkpoint() {
  [[ -s "$CKPT/checkpoint.pt" ]] || { echo "Missing checkpoint: $CKPT/checkpoint.pt (run train)" >&2; exit 1; }
  "$P" - "$CKPT/checkpoint.pt" "$SEED" "${base_files[@]}" "${swap_files[@]}" <<'PY'
import sys, torch
cfg = torch.load(sys.argv[1], map_location="cpu", weights_only=False)["args"]
assert cfg["diff_steps"] == 100, f"Expected trained diffusion schedule=100, got {cfg['diff_steps']}"
assert cfg["horizon"] == 32 and cfg["steps"] == 60000
assert set(cfg["modes"]) == {2, 4} and cfg["norm"] == "minmax" and cfg["clip"] == 1.0
assert cfg["conditioning_version"] == 4
assert cfg["seed"] == int(sys.argv[2]), "Checkpoint seed differs; choose a new root"
assert cfg["data"] == sys.argv[3:], "Checkpoint training data differs; choose a new root"
print("Verified: 100 diffusion levels, 32-state horizon, 60k updates, Mode 2/4")
PY
}
preflight() {
  echo "Running existing regression tests (must pass before GPU training)..."
  "$P" -m pytest -q tests/test_joint_diffusion_guidance.py tests/test_mode4_anchor.py tests/test_tt_rollout_loader.py tests/test_tt_physics_semantics.py tests/test_v7_ddim100_audit.py
  require_base
  [[ -d "$EXPERT" ]] || { echo "Missing expert config: $EXPERT" >&2; exit 1; }
}
prepare() {
  require_base
  if [[ -s "${swap_files[0]}" ]]; then
    echo "Reusing existing E-swap files where present; collector skips complete targets."
  fi
  "$P" code/tt_dynamics/collect_e_swap.py --out "$SWAP" --sources "${designs[@]}" --targets "${designs[@]}" --episodes 32 --steps 100 --seed "$SEED"
  require_data
}
train() {
  require_data
  if [[ -e "$CKPT/checkpoint.pt" ]]; then
    echo "Existing checkpoint retained: $CKPT"
    require_checkpoint
    return
  fi
  [[ ! -d "$CKPT" ]] || { echo "Partial training dir exists; refusing overwrite: $CKPT" >&2; exit 1; }
  echo "TRAIN: new TT from scratch, 100 diffusion steps, 60,000 optimizer steps."
  "$P" code/tt_dynamics/train_tt.py \
    --data "${base_files[@]}" "${swap_files[@]}" \
    --out "$CKPT" --horizon 32 --steps 60000 --batch 32 \
    --eval-episodes 8 --modes 2 4 --diff-steps 100 \
    --norm minmax --clip 1 --sampler ddim --seed "$SEED" \
    > "$ROOT/train.log" 2>&1 || { tail -n 60 "$ROOT/train.log" >&2; exit 1; }
  require_checkpoint
}
generate() {
  local out="$1"; shift
  "$P" code/tt_dynamics/eval_v7_ddim100.py --generate "$P" code/tt_dynamics/generate_task.py \
    --ckpt "$CKPT" --out "$out" --expert-config "$EXPERT" \
    --anchor-s0 --joint-full-guidance --e-rest-s0 --no-s0-align \
    --seed "$SEED" "$@"
}
smoke() {
  require_checkpoint
  generate "$ROOT/smoke_zigzag_2s" --no-render --task zigzag --seconds 2 \
    --candidates "${V7_100_SMOKE_CANDIDATES:-8}" --scales 0 1 3 10
  "$P" code/tt_dynamics/eval_v7_ddim100.py --root "$ROOT" --stage smoke --require-smoke-gate
}
codesign() {
  require_checkpoint
  # Same fixed-E comparison design as v6; e003 remains unseen.
  for task in zigzag circle; do
    generate "$ROOT/codesign/fixedE_${task}_4s" --no-render --task "$task" --seconds 4 \
      --candidates "${V7_100_FIXED_PER_E:-4}" --scales "$FIXED_SCALE" --fixed-e "${designs[@]}"
    generate "$ROOT/codesign/joint_${task}_4s" --task "$task" --seconds 4 \
      --candidates "${V7_100_JOINT_CANDIDATES:-16}" --scales "$FIXED_SCALE"
  done
  "$P" code/tt_dynamics/eval_v7_ddim100.py --root "$ROOT" --stage codesign --fixed-scale "$FIXED_SCALE"
}
full() {
  require_checkpoint
  # 100 denoising steps x ~16 windows is expensive: start with ONE 10s candidate.
  # Inference stays 10-second OPEN-LOOP, without replanning or PPO fine-tuning.
  for task in zigzag circle; do
    generate "$ROOT/full_${task}_10s" --no-render --task "$task" --seconds 10 \
      --candidates "${V7_100_FULL_CANDIDATES:-1}" --scales 0 1 3
  done
  "$P" code/tt_dynamics/eval_v7_ddim100.py --root "$ROOT" --stage full
}
case "$stage" in
  prepare) preflight; prepare;;
  train) preflight; train;;
  smoke) smoke;;
  codesign) codesign;;
  full) full;;
  report) "$P" code/tt_dynamics/eval_v7_ddim100.py --root "$ROOT" --stage all --fixed-scale "$FIXED_SCALE";;
  all)
    preflight
    prepare
    train
    smoke
    codesign
    full
    "$P" code/tt_dynamics/eval_v7_ddim100.py --root "$ROOT" --stage all --fixed-scale "$FIXED_SCALE"
    ;;
esac
echo "Finished stage $stage. Outputs: $ROOT (NOT a claim of task success)."
