# V100: CALM 10,000 → 200 morphologies → CPU MuJoCo dynamics → TT → Reward Gradient

This is a **new, isolated, speed-first experiment**, not a modification of the original
shared-PPO/MJX experiment. Base branch: `cleanup/shared-ppo`. Implementation branch:
`experiment/v100-mujoco-10k-tt`.

## Scientific contract

Question: does broadening the dynamics data from the former small motion/morphology
dataset improve *generated morphology* and *physical composite-task execution*?

* Original accepted CALM references: 10,000, verified by `dataset/status.json`.
* Train morphology: 200 A1 topology `00000011`, five real size parameters varied
  within ±20%; 20 separate held-out morphologies. No changes to knee topology.
* Assignment: each motion assigned to two training morphologies, 100 motions/design;
  collect two starting conditions per pair: **40,000 planned training episodes**.
  Held-out morphologies remain in `heldout/` and are *never* given to TT.
* Physics: **CPU MuJoCo `mujoco.MjData/mj_step`** with a 5 ms physics step and
  position actuators (kp=30, torque limit ±33.5 N m), target_q 50 Hz.
  Retarget CALM 30 Hz references to matching E, interpolate joint targets to 50 Hz.
* Fastest initial collection: **scripted reference-joint PD tracking, no PPO**.
  This is **not** the original per-primitive PPO, transition expert, or shared PPO.
  Tracking quality and falling rates MUST be reported. It may fail often and is
  not guaranteed to reach 40,000 usable episodes; failures remain in the dataset.
  This alternative collector changes the original expert-data pipeline and therefore
  is a separate experiment, not evidence about PPO tracker performance.
* TT: `code/tt_dynamics/train_tt.py`, reward-agnostic joint E,S,A diffusion,
  horizon 32, mode 1/2/4 (0.2/0.4/0.4), 5 denoise steps for speed,
  25k optimizer updates, 256 width, six layers, minmax normalization, 80/20
  split by original motion `group_id` across all morphologies; TT uses
  **absolute target joint angles** and the real physical qpos/qvel/contact.
  Models use the existing TT 50 Hz / 500 step episode assumption.
* Guided generation: same frozen TT, same noise seeds for guidance scale 0 vs 1,
  **full 10-second** morphology E + state S + target actions A, three tasks:
  `zigzag`, `circle`, `straight` (maximize forward progress over 10 s,
  with stability, lateral drift and action regularization).
* Evaluation: **independent CPU MuJoCo** physical open-loop 10 s replay,
  output actual reward, distance, falls, initial-state validity and trajectories,
  not merely the TT predicted reward.
* Fast mode is an exploratory configuration. Do NOT claim that 5 denoising steps,
  PD tracking, collection completion, or 25k updates imply accurate dynamics,
  successful control, or convergence.

## Prerequisite

From V100 SSH, have the existing `~/quad_ws/codesign` and a working
`../transformer-transformer/.venv/bin/python` as described in `SETUP.md`.
Install *only* from the repository's existing pinned requirements. Retargeting
uses `mink` and `scipy`; generation uses `torch` and `mujoco`.
V100 CUDA is used for **PyTorch TT**, not physical simulation.

First obtain this branch (do not overwrite uncommitted work):

```bash
cd ~/quad_ws/codesign
git status --short
git fetch origin
git switch -c experiment/v100-mujoco-10k-tt --track origin/experiment/v100-mujoco-10k-tt
export PYTHON="$PWD/../transformer-transformer/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
"$PYTHON" -c 'import mujoco,torch; print("MuJoCo",mujoco.__version__,"PyTorch GPU",torch.cuda.is_available())'
```

Important: if the branch is already checked out locally, use
`git switch experiment/v100-mujoco-10k-tt` after fetching; do not run `-c`
twice. The older V100 run outputs are never used as the `--out` path.

## Initial health check (NOT the main experiment)

Create a new plan. `init` rejects an existing plan; no overwrite:

```bash
"$PYTHON" code/motion_tracking/collect_calm_mujoco.py init \
  --out runs/v100_mujoco_10k
```

Run one morphology **only** to verify the retarget → MuJoCo → NPZ path:

```bash
"$PYTHON" -u code/motion_tracking/collect_calm_mujoco.py run \
  --out runs/v100_mujoco_10k --limit-designs 1 --workers 1
"$PYTHON" code/motion_tracking/collect_calm_mujoco.py audit \
  --out runs/v100_mujoco_10k
```

This pilot is a *subset of the exact same immutable full plan*, and the
runner resumes previously completed per-design output. Inspect
`audit.json`, including `tt_usable_episodes`, `valid_transitions`,
`falls`, `source_coverage`, `invalid_files`. Pilot audit `complete=false`
is expected. Do not mistake source motion coverage for successful tracking.

## Full collection

```bash
nohup "$PYTHON" -u code/motion_tracking/collect_calm_mujoco.py run \
  --out runs/v100_mujoco_10k --workers 8 \
  > runs/v100_mujoco_10k/collection.log 2>&1 < /dev/null &
echo $! > runs/v100_mujoco_10k/collection.pid
```

This uses 8 independent CPU processes and resumes any already saved design.
For machines with fewer CPUs or memory constraints reduce `--workers`.
Monitoring:

```bash
tail -f runs/v100_mujoco_10k/collection.log
"$PYTHON" code/motion_tracking/collect_calm_mujoco.py audit --out runs/v100_mujoco_10k
```

The reference dataset has 10 s and 20 s sources. Both are sampled into
up to 10 s / 500-step episodes at 50 Hz. The two start variants are distinct
time windows; they are not matched action-intervention pairs.
Datasets and model outputs are LOCAL, generally much larger than GitHub's
100 MB/file limit. Do not add the complete `runs/v100_mujoco_10k/` directory
to Git; upload only compact audit/summary and reproducible scripts if needed.

## TT training (only after full audit complete)

```bash
"$PYTHON" code/motion_tracking/collect_calm_mujoco.py audit --out runs/v100_mujoco_10k
nohup "$PYTHON" -u code/motion_tracking/collect_calm_mujoco.py train \
  --out runs/v100_mujoco_10k --steps 25000 --batch 64 \
  > runs/v100_mujoco_10k/train_launcher.log 2>&1 < /dev/null &
```

The train command rejects an incomplete audit and will NOT silently use a
4-motion pilot or the old 50 Hz rollouts. The actual training config and
split audit appear under `runs/v100_mujoco_10k/tt/h32_fast/`.
The runner writes all output to `tt_train.log`.

### TT completion vs convergence

**Finished successfully** means *all* of the following are true:

1. `audit.json` has `complete: true`, 10,000 original sources, 200
   training morphologies, all 40,000 planned saved episodes, nonzero usable
   transitions, no invalid files.
2. `tt/h32_fast/checkpoint.pt` exists and can be loaded; the checkpoint
   configuration names data from `runs/v100_mujoco_10k/rollouts/`,
   `modes=[1,2,4]`, `diff_steps=5`, `horizon=32`.
3. `tt/h32_fast/progress.jsonl` last step = 25000 and `val_loss` is finite;
   `tt_train.log` has no traceback.
4. `tt/h32_fast/split_audit.json` confirms source-based disjoint
   train/validation source groups, 80/20 group split.

**Convergence and ability to generate valid actions is a different question.**
Those require loss curves and actual MuJoCo composite task evaluation.

## Three physics-validated reward-gradient tests (only after TT completed)

```bash
for task in zigzag circle straight; do
  "$PYTHON" -u code/tt_dynamics/generate_task_mujoco.py \
    --ckpt runs/v100_mujoco_10k/tt/h32_fast \
    --out "runs/v100_mujoco_10k/task_$task" \
    --task "$task" --candidates 4 --scales 0 1 --seed 0
done
```

Each task output contains `summary.json` and `scale_0.npz` /
`scale_1.npz` (predicted and actual trajectory, E, physical validity).
Compare each candidate under *same* noise seed between guidance scales.
The generated morphology is not preselected from training shapes.
Review predicted-vs-actual score and physical validity; invalid S0 does
**not** count as successful reward 0.

NOTE: this branch uses 5 diffusion steps deliberately to obtain fast
preliminary feedback. Any claim of final TT performance requires a separate,
matched-budget, higher-step training/generation comparison.
