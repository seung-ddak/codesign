# Full local CALM 10k multi-morphology dynamics collection

This supersedes the three-morphology `dynamics_residual_20261109` queue at the
user's request. Saved small-run files remain untouched and are not mixed in.

The imported GitHub implementation is `collect_calm_mujoco.py` from
`experiment/v100-mujoco-10k-tt`, commit `0110311b4`. `provenance.json` records
the full commit and local file hashes. Local changes add per-design timing,
persistent progress/ETA, and stricter finite-value checks. The optional
`flat_ground` model argument is brought over with its original default unchanged.

## Immutable allocation

- All **10,000 accepted original motions**, checked against the local source
  status/journal and source file hashes.
- **200 training morphologies**, varying five physical dimensions within ±20%.
- Every motion assigned to two different training morphologies, 100 motions per
  morphology. Two start variants per pair: **40,000 training episodes**.
- **20 held-out morphologies**, stored separately under `heldout/`: 4,000 additional
  evaluation episodes when `--include-heldout` is used. Held-out data is never a
  TT training input.
- Reference targets start at 30Hz, but actual physics is simulated at **50Hz
  control / 200Hz physics**, kp=30, torque limit ±33.5Nm, flat ground. Do not
  mix the old 30Hz rollout files into this dataset.

The imported collector uses reference-joint PD, not a learned balance policy.
It records actual qpos/qvel/contact and applied absolute joint targets, including
valid failure transitions. This is dynamics collection, not proof of expert
tracking or successful task execution. The upstream run had about 64% falls.

## Execution and limits

First run one complete morphology/200 episodes with one CPU worker. Then use
six workers for all 200 training plus 20 held-out shapes; completed files are
skipped. One BLAS/OpenMP thread per worker; CUDA disabled. Full service RAM
high watermark 8GiB, maximum 12GiB, no swap, CPU quota 600%. This collector does
not allocate GPU memory. Saved episode files use atomic writes.

The full service is `local-mujoco-10k-20261109`; the pilot is
`local-mujoco-10k-pilot-20261109`. `collection_progress.json` reports wall time,
completed shapes, errors and ETA. `audit.json` is the final training-source
coverage/validity gate. A pilot audit with `complete=false` is expected.

Pilot measurement: 153.2 seconds for 200 episodes/61,802 valid transitions;
125 falls, no invalid files. The six-worker full run started at about 18:35 KST.
Linear projection from this one shape is about 20:10 KST; allow until 21:00–22:00
for morphology differences and contention. The service stops after five hours
(about 23:35 KST) if unfinished, retaining completed outputs. Resume is required
after a timeout. `status.json` is the initialization snapshot; use the live
`collection_progress.json` and systemd service state to assess current execution.

```bash
systemctl --user status local-mujoco-10k-20261109
tail -n 10 runs/local_mujoco_10k_20261109/collection.log
cat runs/local_mujoco_10k_20261109/collection_progress.json
```

Resume collection only after the previous service has stopped:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' MUJOCO_GL=egl \
../transformer-transformer/.venv/bin/python -u code/motion_tracking/collect_calm_mujoco.py \
  run --out runs/local_mujoco_10k_20261109 --workers 6 --include-heldout
```

For completion, verify all 10k source motions, all 200 training shapes, 40k
saved training episodes, no invalid files, and usable transition coverage. Also
check all 20 held-out shapes have both variants. Collection completion is not
equivalent to accurate dynamics predictions or control success.

## Subsequent TT and residual RL

2026-11-10 순차 비교 설계: [공통 TT → reward-gradient → RL 잔차 체인](../../docs/TT_REWARD_GRADIENT_RL_CHAIN.md). 설계 문서이며 실행기와 원본 데이터 어댑터는 아직 구현 전이다.

No automatic TT training is scheduled. The imported collector's `train` command
describes the upstream speed-first H64/five-step experiment, **not** the user's
intended original-TT reproduction. Do not use that command for the next phase.
Use the original `t2` token adapters/DiT/AdaLN/normalization and task schedule
after checking their baseline configuration. Split original source hashes across
all morphologies and start variants; never split overlapping windows separately.

Freeze TT for residual RL and add a bounded residual at each original sampler
step in its actual prediction parameterization. Zero residual must reproduce
the original sampler. Use actual physical task return for PPO; report model
error separately. Avoid the previous cached one-step TT reconstruction and
predicted-return-only selection. This may improve actual execution despite model
error; only physical evaluation can establish improvement.
