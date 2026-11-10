#!/usr/bin/env bash
# Collect wide16 expert actions, replay Orig/Swap/Offset, then train/evaluate matched TT models.
set -euo pipefail
cd "$(dirname "$0")/../.."
export TMPDIR="$PWD/.runtime/tmp" MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0
mkdir -p "$TMPDIR"
python=../transformer-transformer/.venv/bin/python
root=runs/pipeline_v1/wide16_action_shared
rollout_root=runs/pipeline_v1/wide16_action/rollouts
expert=runs/pipeline_v1/experts_wide16_v2
read -r -a designs <<< "${WIDE16_DESIGNS:-e000 e001 e002 e004 e005 e006 e007 e008 e009 e010 e011 e012 e013 e014 e015 e016}"
experts=("$expert"/*__r00)
[[ ${#experts[@]} -eq 19 ]] || { echo "Expected 19 wide16 experts" >&2; exit 1; }
mkdir -p "$root"
"$python" - "$root/episode_split.json" <<'PY'
import json, pathlib, sys
import numpy as np
p = pathlib.Path(sys.argv[1])
permutation = np.random.default_rng(0).permutation(304)
expected = {"seed": 0, "episodes": {str(int(e)): ("train" if i < 213 else "val" if i < 259 else "test")
                                     for i, e in enumerate(permutation)}}
if p.exists():
    assert json.loads(p.read_text()) == expected, "shared episode split changed"
else:
    p.write_text(json.dumps(expected, indent=2) + "\n")
assert len(expected["episodes"]) == 304 and {v: list(expected["episodes"].values()).count(v)
                                                  for v in ("train", "val", "test")} == {"train": 213, "val": 46, "test": 45}
PY

all=() orig=()
for design in "${designs[@]}"; do
  rollout="$rollout_root/$design/primitive_seed0.npz"
  dir="$root/$design"
  mkdir -p "$dir"
  if [[ ! -f "$rollout" ]]; then
    "$python" code/motion_tracking/collect_rollouts.py "${experts[@]}" --design "$design" \
      --mode primitive --episodes 304 --out "$rollout_root" > "$dir/rollout.log" 2>&1
  fi
  "$python" code/motion_tracking/check_rollouts.py "$rollout" >> "$dir/rollout.log"
  if [[ ! -f "$dir/initial.npz" ]]; then
    "$python" code/tt_dynamics/collect_action_pairs.py --source "$rollout" --design "$design" \
      --expert-config "$expert/walkForwardSlow__r00" --starts 192 --out "$dir/initial.npz" > "$dir/initial.log" 2>&1
  fi
  "$python" code/tt_dynamics/prepare_ablation.py --source "$dir/initial.npz" --out-dir "$dir" \
    --episode-split "$root/episode_split.json"
  if [[ ! -f "$dir/paired.npz" ]]; then
    "$python" code/tt_dynamics/collect_action_pairs.py --starts-from "$dir/initial.npz" \
      --manifest "$dir/split_manifest.json" --design "$design" \
      --expert-config "$expert/walkForwardSlow__r00" --out "$dir/paired.npz" > "$dir/paired.log" 2>&1
  fi
  if [[ ! -f "$dir/model_a_orig.npz" && ! -f "$dir/model_b_all.npz" ]]; then
    "$python" code/tt_dynamics/prepare_ablation.py --source "$dir/initial.npz" \
      --out-dir "$dir" --paired "$dir/paired.npz" --episode-split "$root/episode_split.json"
  fi
  [[ -f "$dir/model_a_orig.npz" && -f "$dir/model_b_all.npz" ]] || { echo "Incomplete pair data: $dir" >&2; exit 1; }
  orig+=("$dir/model_a_orig.npz")
  all+=("$dir/model_b_all.npz")
done

out_b="$root/tt_all"
out_a="$root/tt_orig"
if [[ ! -f "$out_b/checkpoint.pt" ]]; then
  [[ ! -d "$out_b" ]] || { echo "Incomplete TT directory: $out_b" >&2; exit 1; }
  "$python" code/tt_dynamics/train_tt.py --data "${all[@]}" --out "$out_b" --horizon 32 \
    --steps 20000 --batch 32 --modes 1 --diff-steps 5 --norm minmax --clip 1 \
    --sampler ddim --split-key split > "$root/train_all.log" 2>&1
fi
if [[ ! -f "$out_a/checkpoint.pt" ]]; then
  [[ ! -d "$out_a" ]] || { echo "Incomplete TT directory: $out_a" >&2; exit 1; }
  "$python" code/tt_dynamics/train_tt.py --data "${orig[@]}" --out "$out_a" --horizon 32 \
    --steps 20000 --batch 32 --modes 1 --diff-steps 5 --norm minmax --clip 1 \
    --sampler ddim --split-key split --norm-from "$out_b/checkpoint.pt" > "$root/train_orig.log" 2>&1
fi
for design in "${designs[@]}"; do
  "$python" code/tt_dynamics/eval_action_pairs.py --data "$root/$design/model_b_all.npz" \
    --old "$out_a" --new "$out_b" --split test --out "$root/$design/comparison.json" \
    --plot "$root/$design/comparison.png" > "$root/$design/eval.log" 2>&1
done
for trio in 'e000 e001 e002' 'e000 e008 e016'; do
  read -r -a shapes <<< "$trio"
  if [[ -f "$root/${shapes[0]}/split_manifest.json" && -f "$root/${shapes[1]}/split_manifest.json" && -f "$root/${shapes[2]}/split_manifest.json" ]]; then
    for model in orig all; do
      ckpt="$out_b"
      [[ "$model" == orig ]] && ckpt="$out_a"
      out="$root/d4_${shapes[0]}_${shapes[1]}_${shapes[2]}_${model}"
      if [[ ! -e "$out" ]]; then
        "$python" code/tt_dynamics/d4_morphology.py --ckpt "$ckpt" --out "$out" \
        --shapes "${shapes[@]}" --rollouts-root "$rollout_root" --rollout-name primitive_seed0.npz \
        --expert-config "$expert/walkForwardSlow__r00" --align-height \
        --manifests "$root/${shapes[0]}/split_manifest.json" "$root/${shapes[1]}/split_manifest.json" \
                    "$root/${shapes[2]}/split_manifest.json" > "$root/d4_${shapes[0]}_${shapes[1]}_${shapes[2]}_${model}.log" 2>&1
      fi
      [[ -f "$out/d4_morphology.json" ]] || { echo "Incomplete D4 evaluation: $out" >&2; exit 1; }
    done
  fi
done
echo "Completed wide16 action comparison: $root"
