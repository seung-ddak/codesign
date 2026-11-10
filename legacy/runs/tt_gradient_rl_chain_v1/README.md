# Original TT / reward-gradient / residual PPO chain

Implementation: `code/tt_dynamics/run_native_tt_chain.py`. Source data is the
completed `runs/local_mujoco_10k_20261109` collection. The full chain uses the
original t2 DiT-small, modality adapters, DDIM 100-step task and EMA. It does
not call the earlier simplified H64/five-step TT trainer.

The stages are **preflight → native data preparation → common TT pretraining →
gradient development evaluation → residual PPO training → final comparison**.
Each runs in a separate subprocess. Failure stops the chain. Reward-gradient
does not update TT weights. PPO learns per-token epsilon residuals with TT
frozen, using actual physical return, without the old predicted-state penalty.

## Commands

From the codesign root, the underlying resumable command is:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/chain_mpl \
../transformer-transformer/.venv/bin/python -u \
  code/tt_dynamics/run_native_tt_chain.py run --device cuda
```

The managed execution is the user service `native-tt-chain-v1`, with RAM
MemoryHigh=8GiB/MemoryMax=12GiB, no service swap, CPU quota 400%, and one stage
at a time. GPU stages cap the PyTorch allocator at 75% of VRAM (about 9GiB).
Other CUDA/OpenGL allocations are not governed by that allocator setting.

```bash
systemctl --user status native-tt-chain-v1
cat runs/tt_gradient_rl_chain_v1/chain.json
cat runs/tt_gradient_rl_chain_v1/data/progress.json
tail -n 3 runs/tt_gradient_rl_chain_v1/tt_base/progress.jsonl
```

Do not run another instance alongside the service. After an interruption,
resume the same code/config/source with:

```bash
systemctl --user restart native-tt-chain-v1
```

Completed stages are checked before being skipped. Conversion commits one
shard atomically. Training snapshots contain optimizer/scheduler/EMA and all
RNG state; TT snapshots are written every ten updates and at the end, PPO
snapshots every update. Up to nine TT updates can be repeated after an abrupt
stop. Source and runtime hashes protect against resuming a different experiment.
Code/config changes require preserving the old run and deliberately creating a
new compatible run; do not delete checkpoints to get past a contract failure.

## Validation and limits

- Actual source motor targets survive normalization/token round-trip.
- Original native loss forward/backward passes on real collected data.
- A zero-mean residual reproduces the original 100-step sampler; nonzero
  residual changes the commands passed to physics.
- Fixed held-out hardware is inpainted, including native padding, and checked
  for unchanged decoded hardware values. This conditioning is an explicit
  inference adaptation; it is not an independently trained conditional task.
- Physical reward is reproduced by independent reruns and checked against
  recorded per-step reward sums. Invalid generated graphs receive -20 and
  are reported. They are never replaced with a training robot.
- Installed PPO learns a known target residual in a small test. A separate
  16-candidate × 100-step update checks actual sampler/PPO integration.
- `resume_check.json` compares interrupted versus uninterrupted actual TT
  updates with effective batch 1,024; observed parameter difference was zero.

The tokenizer's geometry/inertia split is checked per shape against the source
model's mass matrix (1% maximum relative difference). The added tracking token
is on the rigid quadruped trunk; the removed manipulator is not restored.
Training observations use original-model actuator forces/velocities rather
than dynamics from the tokenization copy. This adaptation and measured mass
matrix error are recorded; it is not a claim of bit-exact physical identity.

Native training samples eight distinct times from each actual episode, as in
the original RandomTimestepSampler. All 40,000 collected episodes have at least
11 valid steps, so all sources/episodes can participate. The stock loader's
100-step minimum is deliberately not applied: valid short failure episodes
are preserved. Source hashes, not windows or start variants, define the 80/20
split. Validation/held-out samples never determine normalization.

Generation uses eight native timestamps spanning a 10-second task; absolute
targets are linearly interpolated to 500 actual control steps. This temporal
interpolation is an explicit common adaptation for U/G/R, not a learned dense
500-step trajectory or the old ad hoc H64 stitching. Full hardware tokens are
generated/validated natively rather than substituted with a nearest training E.

Budget: original TT 20,480 optimizer updates, effective batch 1,024; PPO three
tasks × three seeds × 1,000 updates × 16 physical episodes. Actual throughput
and ETA appear in `tt_base/progress.jsonl` and each policy's logs. The smaller
batch-four pilot took about 7.3 seconds/update (roughly 42 hours for TT alone);
the memory calibration may select a larger microbatch. This estimate excludes
full-data I/O and does not promise convergence or RL task success.

Final free-hardware evaluation uses 64 held-out noise seeds per task/training
seed/method. A separate fixed-hardware evaluation uses one identical noise
per task/seed for each of the 20 held-out shapes. Guidance scale and PPO
checkpoint selection use development seeds only. If scale 0 wins, the report
must be interpreted as development selection preferring no gradient guidance.

Small-run artifacts under `pilot_*` and `resume_check_*` are connection tests;
their weights/data are not used to initialize the full TT or actual RL policies.
