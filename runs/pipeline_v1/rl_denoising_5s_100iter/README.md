# Diffusion PPO: 5 seconds, 8 candidates, 100 iterations

2026-11-03. Latest user-requested settings:

- Jointly generate E, future S, A from eight distinct noise samples each iteration; E is not fixed across iterations.
- Frozen v16 twin TT, five reverse-diffusion decisions per candidate.
- 250 state frames at 50Hz: 249 physical control transitions (4.98 seconds), displayed as five-second videos. Falling ends a rollout early.
- 100 PPO collection/update iterations, four optimizer epochs per update; seed0.
- Per task: 800 training physical episodes, 4000 PPO denoising transitions. Physics control steps are separate from PPO transitions.
- Zigzag and circle use separate PPO policies. Batch size8, existing dense physical reward minus 5×predicted/physical position MSE, unchanged.
- Initial S0 is rest(E), future S/A are generated. TT weights remain frozen. PPO corrects epsilon for E and window-level S/A channels; no reward gradient during RL sampling.
- CPU MuJoCo. Deterministic evaluation uses held-out initial-noise seed1000000 and eight candidates; train noise seeds0–99. Evaluation policies are unguided, reward-gradient(scale10), PPO before/after training. No checkpoint selection using evaluation results.

The earlier 20-second run was stopped after one update per task when the user changed the duration to five seconds. Its artifacts are preserved under `../rl_denoising_100iter/`; they are not used here. The old 200-frame, two-update smoke report remains under `../rl_denoising_v1/`.

Run from workspace root, using a fresh output directory:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl transformer-transformer/.venv/bin/python -u codesign/code/tt_dynamics/train_diffusion_rl.py --task zigzag --updates 100 --batch 8 --episode-states 250 --out codesign/runs/pipeline_v1/rl_denoising_5s_100iter/zigzag_seed0
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl transformer-transformer/.venv/bin/python -u codesign/code/tt_dynamics/train_diffusion_rl.py --task circle --updates 100 --batch 8 --episode-states 250 --out codesign/runs/pipeline_v1/rl_denoising_5s_100iter/circle_seed0
```

Each task saves policy.pt, protocol.json, training.jsonl, source snapshots, and generated E/S/A plus recorded physical trajectories. Report videos show the first two evaluation candidates per task, chosen by index, without performance selection. All eight evaluation candidates remain in summary.json and NPZ files.

Limitations: one seed, one held-out noise batch, and a short task duration. More iterations alone do not establish convergence or task success. Policy actions are channelwise constant epsilon offsets within each window; observation uses pooled latent statistics.
