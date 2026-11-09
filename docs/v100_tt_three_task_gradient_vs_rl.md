# TT 10-second task benchmark: Reward Gradient vs episodic PPO

**Existing checkpoint only; TT weights and the six-step research pipeline stay unchanged.**
This is **task inference** using frozen TT parameters, not re-training TT or
PPO motion/transition experts. The physical evaluator is MuJoCo, not MJX.

## Tasks and metrics

| Task flag | Reward class | What "success" actually means |
|---|---|---|
| circle | TrackReward | counterclockwise progress along radius-1m circular track, lane/stability/effort penalties |
| zigzag | ZigzagReward | forward distance following triangular zigzag centerline, lane/stability/effort penalties |
| straight | StraightReward | maximize +X progress (fast forward) while staying in a straight corridor |

Each method generates a normalized 13-component morphology E, the 500-step
predicted 42D physical state and 12D target joint action. The **same**
generated E, S0, A run in physical MuJoCo (PD kp=30, torque=33.5,
control step 0.02s, integrator step 0.005s) without per-step corrections.
Predictions do not count as task success until physically replayed.

Methods:
- **Gradient scale=0**: frozen TT with no reward guidance (negative control).
- **Gradient scale=1**: existing differentiable reward guidance through 5 DDIM
  steps, modifying shared E and S/A latents.
- **RL (episodic PPO)**: frozen TT; a diagonal Gaussian low-dimensional policy
  chooses one E/S/A residual correction vector per episode, broadcast into
  reverse diffusion. Policy is updated with **actual MuJoCo task returns** via
  clipped PPO. Evaluation uses the policy mean and new held-out random seeds.
  It is *not* a reactive walking policy or full per-denoising-step PPO.

All three methods use the same evaluation seed(s), candidate counts,
reward functions and physics. PPO's *extra* training rollouts are logged
separately as its optimization budget; they are **not** included in evaluation
scores. A small PPO budget does not imply convergence.

## Run safely on the server while the UNet job uses the V100

**Never terminate or disturb the existing UNet process.** Explicitly disable
CUDA for this benchmark; TT inference and MuJoCo playback then use CPU.

```bash
cd ~/quad_ws/codesign
git switch experiment/v100-mujoco-10k-tt
git pull --ff-only origin experiment/v100-mujoco-10k-tt
export PYTHON="$PWD/../transformer-transformer/.venv/bin/python"
export CUDA_VISIBLE_DEVICES=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

"$PYTHON" -m pytest -q tests/test_tt_task_benchmark.py
```

### Pilot: all 3 tasks × 2 guidance strengths × RL (small budget)

```bash
"$PYTHON" -u code/tt_dynamics/benchmark_task_inference.py \
  --ckpt runs/v100_mujoco_10k/tt/h64_warmup_ema_mode0124 \
  --out runs/v100_mujoco_10k/task_gradient_vs_ppo_pilot_v1 \
  --device cpu --cpu-threads 2 \
  --tasks circle zigzag straight \
  --methods gradient rl \
  --scales 0 1 --candidates 2 --eval-seeds 1 \
  --rl-updates 2 --rl-rollouts 4 --ppo-epochs 4 \
  2>&1 | tee runs/v100_mujoco_10k/task_gradient_vs_ppo_pilot_v1.log
```

### Larger comparison (distinct output directory)

```bash
"$PYTHON" -u code/tt_dynamics/benchmark_task_inference.py \
  --ckpt runs/v100_mujoco_10k/tt/h64_warmup_ema_mode0124 \
  --out runs/v100_mujoco_10k/task_gradient_vs_ppo_full_v1 \
  --device cpu --cpu-threads 2 \
  --tasks circle zigzag straight \
  --methods gradient rl \
  --scales 0 1 --candidates 4 --eval-seeds 3 \
  --rl-updates 40 --rl-rollouts 4 --ppo-epochs 4 \
  2>&1 | tee runs/v100_mujoco_10k/task_gradient_vs_ppo_full_v1.log
```

Important: TT forward+backward on CPU may consume nontrivial memory and compute
and slow any concurrent CPU-using job. Do the small pilot first and monitor
`top`, `free -h`, and `nvidia-smi`. GPU 0 should not be newly occupied.

If interrupted: repeat the **same** command, adding `--resume`. RL policy
checkpoint and candidate evaluation files are persisted per task; completed
evaluation JSON files are skipped. If interruption leaves only a partial NPZ
without JSON, inspect it rather than deleting prior data blindly.

## Result files

```text
runs/v100_mujoco_10k/task_gradient_vs_ppo_full_v1/
  experiment_config.json
  circle/
    gradient_scale_0/seed_*.json, seed_*.npz
    gradient_scale_1/seed_*.json, seed_*.npz
    rl_training/training.jsonl, policy_state.pt
    rl_ppo/seed_*.json, seed_*.npz
  zigzag/    [same]
  straight/  [same]
  report.json
  report.md
  report.html
```

Each NPZ stores `E`, `pred` (TT states/actions), `mujoco` (actual
states/actions), and `valid` (physical time masks). Candidate JSON also
contains physical reward terms, real 10s completion, duration, forward
distance, progress, morphology, and predicted-vs-replayed state errors.

To regenerate the reports without loading the TT checkpoint, GPU or MuJoCo:

```bash
"$PYTHON" code/tt_dynamics/tt_task_report.py \
  --out runs/v100_mujoco_10k/task_gradient_vs_ppo_full_v1
```

## Interpretation rules

A method succeeds **only if physical MuJoCo scores and completion improve**.
The first metric is 10s completion rate with all attempts in denominator;
then compare physical reward, valid episode duration, track progress, and
morphology. Invalid initial conditions appear separately and are not given
fake zero rewards. The PPO training uses -100 as a learning penalty for
invalid episodes; reporting omits these from the mean valid replay reward.
An early physical fall receives -20, matching the prior experimental task
script. The predicted-vs-physical XY and joint errors help diagnose the TT
model's previously measured dynamics inconsistency.

**No actual three-task results are contained in this commit.** They will only
appear after executing the run against the pre-trained checkpoint and
full MuJoCo environment on the research machine.
