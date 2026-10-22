#!/usr/bin/env bash
# Small end-to-end CP-Design verification. NO Foot-Anchored or D4/D5 gate.
# Use existing, actual MJX primitive rollouts and a valid expert environment.
# stages: train | smoke (2 s) | full (10 s). Does not retrain PPO experts.
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:-smoke}
[[ "$stage" == train || "$stage" == smoke || "$stage" == full ]] || {
  echo "usage: bash code/tt_dynamics/run_quick_validation.sh {train|smoke|full}" >&2; exit 2;
}
python=${QUICK_PYTHON:-../transformer-transformer/.venv/bin/python}
root=${QUICK_ROOT:-runs/pipeline_v1/quick_joint}
ckpt=${QUICK_CKPT:-$root/tt_joint}
mkdir -p "$root"
export MUJOCO_GL=${MUJOCO_GL:-egl} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
if [[ ! -f "$ckpt/checkpoint.pt" ]]; then
  # Default to the original pipeline's physically simulated primitive rollouts
  # on three different morphologies. No data regeneration or expert training.
  data=()
  if [[ -n "${QUICK_DATA_DIR:-}" ]]; then
    shopt -s nullglob
    data=("${QUICK_DATA_DIR}"/e*/primitive_seed0.npz)
    shopt -u nullglob
    if (( ${#data[@]} < 2 )); then
      echo "QUICK_DATA_DIR needs at least two per-morphology MJX rollout NPZs: $QUICK_DATA_DIR/e*/primitive_seed0.npz" >&2
      exit 1
    fi
    max=${QUICK_MAX_FILES:-16}
    data=("${data[@]:0:max}")
  else
    for shape in e000 e001 e002; do
      p="runs/pipeline_v1/rollouts/$shape/primitive_seed0.npz"
      if [[ ! -f "$p" ]]; then
        echo "Missing existing MJX rollouts: $p" >&2
        echo "Copy previous runs/pipeline_v1/rollouts/ files here, or use QUICK_DATA_DIR for 16 shapes, or set QUICK_CKPT." >&2
        exit 1
      fi
      data+=("$p")
    done
  fi
  printf 'Quick training data: %s\n' "${data[@]}"
  echo "Training a small smoke joint model (Mode 2,4 only), no PPO training"
  "$python" code/tt_dynamics/train_tt.py --data "${data[@]}" \
    --out "$ckpt" --horizon 32 --steps "${QUICK_STEPS:-3000}" \
    --batch "${QUICK_BATCH:-16}" --eval-episodes 8 \
    --modes 2 4 --diff-steps 5 --norm minmax --clip 1 --sampler ddim \
    > "$root/train.log" 2>&1 || {
      tail -n 35 "$root/train.log" >&2; exit 1;
    }
fi
"$python" - "$ckpt/checkpoint.pt" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
modes = set(ck["args"]["modes"])
assert {2, 4} <= modes, f"Joint gradient sampler requires Modes 2+4, got {sorted(modes)}"
assert ck["args"].get("conditioning_version") == 4, (
    "Mode 4 changed to XY/yaw-only anchor in version 4. "
    "Use a new QUICK_ROOT to retrain; do not reuse old stationary-rest checkpoint."
)
assert ck["args"]["horizon"] >= 2
print("Compatible checkpoint:", sys.argv[1], "modes", sorted(modes))
PY
if [[ "$stage" == train ]]; then
  echo "Checkpoint ready: $ckpt/checkpoint.pt"
  exit 0
fi
expert=${QUICK_EXPERT_CONFIG:-runs/pipeline_v1/experts_wide16/walkForwardSlow__r00}
if [[ ! -d "$expert" ]]; then
  expert=runs/pipeline_v1/experts/walkForwardSlow__r00
fi
if [[ ! -d "$expert" ]]; then
  echo "Expert controller config missing. Set QUICK_EXPERT_CONFIG to a working trained expert directory." >&2
  exit 1
fi
if [[ "$stage" == smoke ]]; then
  seconds=${QUICK_SMOKE_SECONDS:-2}
else
  seconds=10
fi
task=${QUICK_TASK:-zigzag}
out="$root/${stage}_${task}_joint"
if [[ -f "$out/summary.json" ]]; then
  echo "Existing result $out/summary.json: use another QUICK_ROOT or remove only this output after checking it" >&2
  exit 1
fi
echo "TT joint diffusion generation + MJX: stage=$stage task=$task seconds=$seconds"
"$python" code/tt_dynamics/generate_task.py \
  --ckpt "$ckpt" --out "$out" --expert-config "$expert" \
  --anchor-s0 --joint-full-guidance --no-s0-align --no-render \
  --seconds "$seconds" --candidates "${QUICK_CANDIDATES:-1}" \
  --task "$task" --scales 0 "${QUICK_GUIDANCE_SCALE:-1}"
"$python" - "$out/summary.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print("\nQUICK CP-DESIGN REPORT (not a proof of task-optimality)")
for name, candidates in d["runs"].items():
    for i,r in enumerate(candidates):
        print(name, "candidate", i,
              "E", r["E"],
              "predicted R", round(r["pred_total"], 3),
              "MJX R", round(r["mjx_total"], 3),
              "valid S0", r["physically_valid_s0"],
              "survived", r["mjx_fell_step"] == -1,
              "valid seconds", round(r["mjx_valid_s"], 2))
for name,audit in d.get("joint_full_guidance_audit", {}).items():
    print(name, "nonzero E reward gradient:", audit["e_gradient_nonzero"],
          "window count:", audit["windows"])
print("Caution: this is a direct software feasibility check; require real MJX score improvement and physically valid S0 before claiming co-design.")
PY
echo "Result: $out/summary.json"
