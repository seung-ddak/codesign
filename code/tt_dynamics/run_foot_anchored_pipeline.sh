#!/usr/bin/env bash
# LEGACY / EXPERIMENTAL foot-anchored branch, NOT the active FAST Mode-2/4 path.
# Pose-heldout within trained morphologies: no unseen-morphology generalization claims.
# Staged pipeline: prepare -> dynamics -> joint -> guidance. Each stage is resumable.
set -euo pipefail
cd "$(dirname "$0")/../.."
stage=${1:-prepare}
root=${FOOT_ROOT:-runs/pipeline_v1/foot_anchored_v1}
expert_config=${FOOT_EXPERT_CONFIG:-runs/pipeline_v1/experts_wide16/walkForwardSlow__r00}
python=../transformer-transformer/.venv/bin/python
export TMPDIR="$PWD/.runtime/tmp" MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$root"

if [[ "$stage" == prepare ]]; then
  [[ -f "$root/s0.npz" ]] || "$python" code/tt_dynamics/foot_anchored_s0.py \
    --out "$root/s0.npz" --expert-config "$expert_config" --designs "${FOOT_DESIGNS:-32}" --poses "${FOOT_POSES:-8}" \
    > "$root/s0.log" 2>&1
  [[ -f "$root/data/prepare.done" ]] || "$python" code/tt_dynamics/collect_foot_anchored.py \
    --starts "$root/s0.npz" --out "$root/data" --expert-config "$expert_config" --rollout-batch "${FOOT_ROLLOUT_BATCH:-3}" \
    > "$root/collect.log" 2>&1
  echo "Prepared foot-anchored intervention data in $root/data"
  exit
fi

all=("$root"/data/f[0-9][0-9][0-9].npz)
orig=("$root"/data/f[0-9][0-9][0-9]_orig.npz)
morph=("$root"/data/morph_*.npz)
[[ -f "$root/data/prepare.done" && -e "${all[0]}" && -e "${orig[0]}" && -e "${morph[0]}" ]] || { echo "Run prepare first" >&2; exit 1; }

if [[ "$stage" == dynamics ]]; then
  [[ -f "$root/tt_intervention/checkpoint.pt" ]] || "$python" code/tt_dynamics/train_tt.py \
    --data "${all[@]}" --out "$root/tt_intervention" --horizon 32 --steps 20000 --batch 32 \
    --modes 1 --diff-steps 5 --norm minmax --clip 1 --sampler ddim --split-key split \
    > "$root/train_intervention.log" 2>&1
  [[ -f "$root/tt_orig/checkpoint.pt" ]] || "$python" code/tt_dynamics/train_tt.py \
    --data "${orig[@]}" --out "$root/tt_orig" --horizon 32 --steps 20000 --batch 32 \
    --modes 1 --diff-steps 5 --norm minmax --clip 1 --sampler ddim --split-key split \
    --norm-from "$root/tt_intervention/checkpoint.pt" > "$root/train_orig.log" 2>&1
  [[ -f "$root/tt_morph/checkpoint.pt" ]] || "$python" code/tt_dynamics/train_tt.py \
    --data "${all[@]}" "${morph[@]}" --out "$root/tt_morph" --horizon 32 --steps 20000 --batch 32 \
    --modes 1 --diff-steps 5 --norm minmax --clip 1 --sampler ddim --split-key split \
    --norm-from "$root/tt_intervention/checkpoint.pt" > "$root/train_morph.log" 2>&1
  for data in "${all[@]}"; do
    design=$(basename "$data" .npz)
    # Every E has interleaved train/val/test POSES. Do not inspect row zero.
    test_rows=$("$python" -c 'import numpy as np,sys; z=np.load(sys.argv[1]); print(int((z["split"] == "test").sum()))' "$data")
    (( test_rows > 0 )) || continue
    [[ -f "$root/data/${design}_comparison.json" ]] || "$python" code/tt_dynamics/eval_action_pairs.py \
      --data "$data" --old "$root/tt_orig" --new "$root/tt_intervention" --split test \
      --out "$root/data/${design}_comparison.json" --plot "$root/data/${design}_comparison.png"
  done
  # D4 compares within-training-morphology POSE holdouts, not unseen E.
  mapfile -t test_designs < <("$python" - "$root/data" <<'PY'
from pathlib import Path
import numpy as np, sys
for path in sorted(Path(sys.argv[1]).glob("f[0-9][0-9][0-9].npz")):
    with np.load(path) as z:
        if (z["split"] == "test").any():
            print(path.stem)
PY
)
  test_designs=("${test_designs[@]:0:3}")
  [[ ${#test_designs[@]} -eq 3 ]] || { echo "Need three trained morphologies with held-out test poses for D4" >&2; exit 1; }
  echo "D4 tests held-out POSES within trained morphologies, NOT unseen E designs."
  files=(); manifests=()
  for design in "${test_designs[@]}"; do files+=("$root/data/${design}_orig.npz"); manifests+=("$root/data/${design}_manifest.json"); done
  [[ -f "$root/d4_strict_audit/selection.json" ]] || "$python" code/tt_dynamics/d4_morphology.py \
    --ckpt "$root/tt_intervention" --out "$root/d4_strict_audit" --shapes "${test_designs[@]}" \
    --rollout-files "${files[@]}" --manifests "${manifests[@]}" --expert-config "$expert_config" \
    --starts-per-source "${FOOT_D4_STARTS:-8}" --pool-factor 4
  [[ -f "$root/d4_strict_morph/d4_morphology.json" ]] || "$python" code/tt_dynamics/d4_morphology.py \
    --ckpt "$root/tt_morph" --out "$root/d4_strict_morph" --shapes "${test_designs[@]}" \
    --rollout-files "${files[@]}" --manifests "${manifests[@]}" --expert-config "$expert_config" \
    --starts-per-source "${FOOT_D4_STARTS:-8}" --pool-factor 4
  "$python" - "$root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
d4 = json.loads((root / "d4_strict_morph/d4_morphology.json").read_text())
pairs = d4.get("pairs", {})
if not pairs:
    raise SystemExit("no D4 pairs were evaluated")
if not any(int(v) > 0 for v in d4.get("pair_initial_contact_matched_groups", {}).values()):
    raise SystemExit("D4 has no pairwise contact-matched groups")
# Audit only: existence of an evaluable D4 pair is NOT evidence of task-reward
# ranking quality or of adequate dynamics prediction. Require manual review.
(root / "dynamics_gate.json").write_text(json.dumps({
    "passed": False, "reason": "Requires D4/D5 and predicted-vs-MJX task reward review before joint training",
    "strict_d4_pairs": list(pairs),
    "pair_initial_contact_matched_groups": d4.get("pair_initial_contact_matched_groups", {}),
    "checkpoint": d4.get("checkpoint"),
}, indent=2) + "\n")
PY
  echo "Dynamics evaluation complete in $root; D4 shape selection is trained-E/pose-heldout only."
  exit
fi

if [[ "$stage" == joint ]]; then
  "$python" - "$root/dynamics_gate.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file() or not json.loads(p.read_text()).get("passed", False):
    raise SystemExit("Dynamics gate not passed; review D4/D5 and task ranking first")
PY
  # Optional primitive/transition rollouts prepared with compatible E, target_q,
  # valid, and episode-level split arrays (see prepare_general_rollouts.py).
  general=()
  if [[ -d "$root/general_data" ]]; then
    shopt -s nullglob
    general=("$root"/general_data/*.npz)
    shopt -u nullglob
  fi
  echo "Joint data: ${#all[@]} action files, ${#morph[@]} morphology files, ${#general[@]} general files"
  [[ -f "$root/tt_joint/checkpoint.pt" ]] || "$python" code/tt_dynamics/train_tt.py \
    --data "${all[@]}" "${morph[@]}" "${general[@]}" --out "$root/tt_joint" --horizon 32 --steps 20000 --batch 32 \
    --modes 1 4 --diff-steps 5 --norm minmax --clip 1 --sampler ddim --split-key split \
    > "$root/train_joint.log" 2>&1
  [[ -f "$root/joint_32/summary.json" ]] || "$python" code/tt_dynamics/generate_task.py \
    --ckpt "$root/tt_joint" --out "$root/joint_32" --rest-s0 --no-s0-align --no-render \
    --seconds 0.64 --candidates 16 --scales 0
  "$python" code/tt_dynamics/check_joint_generation.py "$root/joint_32/summary.json" --out "$root/joint_gate.json"
  echo "Joint generation passed: $root/joint_gate.json"
  exit
fi


if [[ "$stage" == shared ]]; then
  # Exploratory smoke runs can use FOOT_EXPLORATORY_SHARED=1 without claiming
  # Dynamics passed. Formal runs still require a manually reviewed gate.
  if [[ "${FOOT_EXPLORATORY_SHARED:-0}" != 1 ]]; then
    "$python" - "$root/dynamics_gate.json" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file() or not json.loads(p.read_text()).get("passed", False):
    raise SystemExit("Shared-E formal evaluation requires a reviewed Dynamics gate; exploratory: FOOT_EXPLORATORY_SHARED=1")
PY
  fi
  general=()
  if [[ -d "$root/general_data" ]]; then
    shopt -s nullglob
    general=("$root"/general_data/*.npz)
    shopt -u nullglob
  fi
  [[ -f "$root/tt_joint_shared/checkpoint.pt" ]] || "$python" code/tt_dynamics/train_tt.py \
    --data "${all[@]}" "${morph[@]}" "${general[@]}" --out "$root/tt_joint_shared" \
    --horizon 32 --steps 20000 --batch 32 --modes 1 2 4 5 \
    --diff-steps 5 --norm minmax --clip 1 --sampler ddim --split-key split \
    --norm-from "$root/tt_intervention/checkpoint.pt" > "$root/train_joint_shared.log" 2>&1
  [[ -f "$root/composite_10s_shared/summary.json" ]] || "$python" code/tt_dynamics/generate_task.py \
    --ckpt "$root/tt_joint_shared" --out "$root/composite_10s_shared" \
    --expert-config "$expert_config" --rest-s0 --shared-e-guidance --shared-e-eval-baseline \
    --shared-e-iters "${FOOT_SHARED_ITERS:-4}" --shared-e-perturb "${FOOT_SHARED_PERTURB:-0.04}" \
    --shared-e-step-size "${FOOT_SHARED_STEP:-0.08}" \
    --no-s0-align --no-render --seconds 10 --candidates "${FOOT_SHARED_CANDIDATES:-4}" \
    --scales 0 1 --task zigzag
  echo "Shared-E 10s full-horizon predicted-reward search (not verified by training): $root/composite_10s_shared"
  exit
fi

if [[ "$stage" == guidance ]]; then
  "$python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["passed"], "joint generation gate did not pass"' "$root/joint_gate.json"
  [[ -f "$root/composite_10s/summary.json" ]] || "$python" code/tt_dynamics/generate_task.py \
    --ckpt "$root/tt_joint" --out "$root/composite_10s" --rest-s0 --no-s0-align \
    --seconds 10 --candidates 8 --scales 0 1 5 --task zigzag
  echo "Composite guidance evaluation complete in $root/composite_10s"
  exit
fi

echo "usage: $0 {prepare|dynamics|joint|shared|guidance}" >&2
exit 2
