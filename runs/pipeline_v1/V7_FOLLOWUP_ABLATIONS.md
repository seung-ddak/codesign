# V7 follow-up ablations

Run after the existing V7 E-swap experiment has completed locally. Requires its checkpoint/config, original and E-swap NPZ files, the existing expert config, the sibling transformer-transformer environment, and GPU access. This commit does not include generated datasets, checkpoints, or results.

The runner keeps five diffusion steps to isolate these changes from DDIM100:

1. Re-score V6/V7 saved trajectories without changing historical results.
2. Compare S-gradient multipliers 1.0 and 0.3 using the same V7 checkpoint, seed, 16 candidates, and guidance scale 10, with scale 0 as the reference.
3. Train a separate 60k Mode 2/4 model with 50:50 original/E-swap training minibatches.
4. Train another 60k 50:50 model with Mode 1/2/4 sampling weights 20/40/40. Mode 2/4 exposure decreases; this is not an equal-exposure comparison.

Training preserves V7 source-group train/validation boundaries and reuses V7 normalization. Actual source/mode sample counts and per-source, per-mode validation losses are saved. A two-update GPU implementation check precedes expensive experiments and is not task-performance evidence.

Post-hoc screening separates initial validity, full-duration survival, minimum movement, and task success. The proposed 75% rates, 0.15m path error, and 0.075m/s movement rate are provisional. Zigzag task success additionally requires ordered opposite turns and one period of net progress; circle requires one full forward lap. Both require survival and observed root-corridor compliance. These checks do not require body heading or evaluate the full robot collision envelope.

Each stage records a durable event and posts to issue #8, "Denoising 100 steps", using the existing private `~/.config/codesign/github_token` (mode 600). The token is never included in outputs. Delivery failures preserve pending events and stop unattended execution rather than silently dropping notifications. Numerical screen failures are reported and independent comparisons continue; execution errors are recorded separately. Reports explicitly distinguish completion from task success.

From `codesign/`, use the existing tmux launcher with its unrelated Telegram notifications disabled:

```bash
TMUX_RUN_NOTIFY=0 TMUX_RUN_CWD="$PWD" \
  TMUX_RUN_LOG="$PWD/runs/pipeline_v1/v7_followup_ablation_try1/run.log" \
  bash rl_scripts/tmux_run.sh v7_followup_ablation \
  'MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 ../transformer-transformer/.venv/bin/python code/tt_dynamics/run_v7_ablations.py --out runs/pipeline_v1/v7_followup_ablation_try1 --issue 8'
```

An existing event directory is preserved and rejected: use a fresh output directory for another experiment. Outputs include `report.html`, stage events with delivered comment IDs, motion screens, original-resolution videos, and separate training/evaluation folders. No automatic commit/push of generated artifacts is performed by the experiment runner.
