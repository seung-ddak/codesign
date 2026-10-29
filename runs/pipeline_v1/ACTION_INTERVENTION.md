# Same-start action intervention (e000)

## Question

With morphology E and physical start (qpos, qvel) fixed, does changing the PD joint targets A cause TT to predict the **same change in future state** as MJX? This isolates action conditioning. It does not test morphology generalization or task success.

## Data and comparison

`collect_action_pairs.py` samples valid 32-state windows from the existing e000 rollout. Each start is replayed in MJX three times: original targets, another sampled window's targets, and a constant ±0.2 rad offset per joint. The actual target is clipped to the environment's action range. All three runs share exact starting qpos/qvel, model E and dynamics. It saves `group_id` so training and validation keep the three runs together; falling trajectories remain marked by `valid`, while the evaluator uses only groups whose three runs survive the whole window.

The experiment trains a dynamics-only (`mode 1`: E, s0 and A supplied) TT on these pairs, then compares old and new checkpoints on the **new model's held-out starts**. For each changed A, inspect MJX and TT changes in final x/y, yaw, height and joint angles. A useful result needs lower delta error and a slope nearer 1 on held-out starts, especially for yaw; merely changing predictions is insufficient. A trained-on-pairs result remains in-distribution for one shape and these action perturbations.

Run from `codesign/` with the project's Python environment and an available GPU:

```bash
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0
../transformer-transformer/.venv/bin/python code/tt_dynamics/collect_action_pairs.py \
  --starts 192 --out runs/pipeline_v1/action_pairs/e000_seed0.npz
../transformer-transformer/.venv/bin/python code/tt_dynamics/train_tt.py \
  --data runs/pipeline_v1/action_pairs/e000_seed0.npz \
  --out runs/pipeline_v1/tt/h32_action_pairs --horizon 32 --steps 20000 \
  --diff-steps 5 --norm minmax --clip 1 --modes 1 --sampler ddim
../transformer-transformer/.venv/bin/python code/tt_dynamics/eval_action_pairs.py \
  --data runs/pipeline_v1/action_pairs/e000_seed0.npz \
  --new runs/pipeline_v1/tt/h32_action_pairs --old runs/pipeline_v1/tt/h32/try6 \
  --out runs/pipeline_v1/action_pairs/heldout_d5.json
```

The old checkpoint has three-shape general training while the new checkpoint is trained on one-shape interventions only, so the comparison answers the local action-response question rather than whether the whole generation pipeline improved. The collector will refuse to overwrite existing data.

## First run (2026-10-20)

Host GPU: RTX 4070 SUPER. Collected 192 starts × 3 variants = 576 trajectories; 557 trajectories survived all 32 states. Trained the e000 dynamics-only model for 20,000 steps. Held-out evaluation has 17 starts whose three variants all survived. Full results: `action_pairs/heldout_d5.json`.

| Final-state response to A change | Old TT slope | Paired TT slope | Old/New delta MAE |
|---|---:|---:|---:|
| swap: forward x (m) | 0.395 | 0.660 | 0.112 / 0.083 |
| offset: sideways y (m) | 0.006 | 0.659 | 0.142 / 0.069 |
| swap: yaw (deg) | -0.035 | 0.356 | 11.04 / 10.17 |
| offset: yaw (deg) | 0.044 | 0.619 | 19.49 / 16.78 |
| offset: joint angles (rad) | 0.051 | 0.752 | 0.210 / 0.102 |

These are promising action-sensitivity results on held-out starts, not proof of task success. The old model used a broader three-shape corpus and several conditioning modes, while the new model used only e000 intervention data and mode 1. A controlled ablation with the same corpus/architecture except for paired interventions is needed to attribute the gain specifically to pairing. Shape effects and guided task execution remain untested by this run.

The controlled Orig-only A versus Orig/Swap/Offset B follow-up is recorded in [`action_ablation/README.md`](../action_ablation/README.md).
