# V100 TT: Kinematic consistency warm-start experiment

This is an **ablation**, not a change to the agreed six-stage E/S/A co-design pipeline.
The pretraining remains reward-agnostic. No MuJoCo/PD or 10 s inference files
are modified, and the original checkpoint remains untouched.

## Source and reproducible run

Run on the V100 computer from the repository root:

```bash
cd ~/quad_ws/codesign
git fetch origin experiment/v100-mujoco-10k-tt
git switch experiment/v100-mujoco-10k-tt
git pull --ff-only origin experiment/v100-mujoco-10k-tt
export PYTHON="$PWD/../transformer-transformer/.venv/bin/python"
"$PYTHON" -m pytest -q tests/test_tt_kinematics.py

# First validate which source data/config the runner will use.
"$PYTHON" code/tt_dynamics/run_kinematic_finetune.py \
  --ckpt runs/v100_mujoco_10k/tt/h64_warmup_ema_mode0124 \
  --out runs/v100_mujoco_10k/tt/h64_kinematic_warmstart_v1 \
  --dry-run

# Start 2,000 new gradient updates, initializing model weights from source EMA.
# This is not a resume of the previous 25,000-step optimizer state.
"$PYTHON" -u code/tt_dynamics/run_kinematic_finetune.py \
  --ckpt runs/v100_mujoco_10k/tt/h64_warmup_ema_mode0124 \
  --out runs/v100_mujoco_10k/tt/h64_kinematic_warmstart_v1 \
  --steps 2000 --lr 5e-5 --warmup-steps 100 \
  --kin-xy-weight 0.002 --kin-joint-weight 0.01
```

If interrupted, rerun the **same** command with `--resume`. The runner reads the
source checkpoint's exact ordered `--data` list and matching training contract
(horizon, architecture, diffusion steps, normalization, seed, validation split,
modes, group loss) and rejects mismatches. The new run reuses source
`mu/sd` and source EMA weights, but creates fresh optimizer, schedule, EMA
and training state.

## Loss and observability

In addition to the original epsilon noise-prediction objective, optionally use:
- XY displacement minus trapezoidal heading-rotated velocity integral
  (`dt=0.02 s`, normalized by 2 cm per XY axis).
- Joint angle displacement minus trapezoidal joint-velocity integral
  (normalized by 0.1 rad per joint).

Only neighboring valid frames count. Predicted `x0` is clamped to the same
bounds as denoising and known inpainted conditioning coordinates are kept.
Loss terms are bounded at high diffusion noise by
`min(1, 5*alpha_bar/(1-alpha_bar))` per sample.
**Raw physical RMSE is not Min-SNR weighted.**

See `progress.jsonl`:
- `raw_loss`, `val_raw_loss`: unmodified group-normalized noise errors.
- `loss`, `val_loss`: overall training objective including new losses.
- `xy_kinematic_loss`, `joint_kinematic_loss` and `val_...`: normalized Min-SNR-weighted loss components.
- `xy_kinematic_rmse_cm`, `joint_kinematic_rmse_deg` and `val_...`: raw physical consistency errors.

At completion, check Mode-1 dynamics reconstruction via
`eval_dynamics_ddim.json` (or `eval_dynamics.json` if source sampling is DDPM)
and compare to the original source checkpoint **on the same held-out examples**.
Lower consistency RMSE alone is not a success; actual position/joint prediction
errors must also avoid deterioration. Only then test 10-second physical MuJoCo
rollouts using the unchanged task and seed.

No training was started by this GitHub commit; execute the runner on V100.
