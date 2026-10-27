# CALM reference pipeline: reviewed implementation plan

## Decision and scope

Use the existing CALM HLC/LLC and existing MJX multi-reference tracker. Do not retrain
CALM, build a second tracker, or start 100-reference PPO before source-motion validation.
Worktree/branch: codesign-calm / experiment/calm-reference-pipeline.
Historical codesign runs and existing expert code remain untouched.

## Reuse and required corrections

| Component | Reuse | Required work before training |
|---|---|---|
| CALM HLC | HRLPlayer._compute_llc_action: remove task obs, LLC normalization, normalized latent, action clamp/rescale | Use exact c8 heading config and provided weights; record command before observation construction |
| CALM commands | Existing target velocity/facing fields and training ranges (0..1 m/s) | RANDOM_CMD is piecewise constant, includes posture commands, and keyboard playback resets after done; it does not meet continuous smooth-motion collection requirements |
| Reference conversion | pd_replay.retarget_tt and existing fix_* format | Isaac joint order FL/FR/RL/RR differs from MJCF FR/FL/RR/RL; xyzw -> wxyz; name-map joints; check FK equivalence before resampling; preserve time endpoints and use SLERP |
| Multiple motions, single PPO | TrackMjxEnv already loads multiple clips, samples clips and observes future reference | Pass all stage clips to BOTH train/eval; do not create one PPO per clip |
| RSI | Existing reset_with_traj uses matching pose/velocity | Existing loader accepts every frame; add offline joint-limit/penetration validation and valid-frame sampling before training. Set reset_qpos/reset_qvel=0 for first reference fidelity baseline |
| Alignment | ref_off applies one SE(2) offset | switch_prob=0 for continuous captured clips; do not splice/re-align each step |
| PPO entry point | train_rl_procedural, restore_ckpt | Current defaults multi_design=true; set false for e000. It appends topology to reference_dir; configure parent e000 directory accordingly |
| Evaluation | eval_motions.py and existing rollout collector | Evaluate each clip from frame 0 for full duration, no auto-reset stitching. Report pose/velocity/feet/contact and per-reference metrics separately from reward |
| Hard-clip sampling | clip_boost | Offline evaluation between stages; capped boost, replay mastered clips. No second online sampling subsystem needed initially |
| Curriculum | existing checkpoint restore | Explicit stage clip lists of 20/50/100 and resume same network; gate from fixed metrics, not update counts. Define/freeze thresholds after Stage-1 diagnostics, before evaluating later stages |

## Current bounded verifier

`code/motion_tracking/calm/verify_hlc.py` runs one original A1 environment, Default
initialization, deterministic HLC and original LLC, native 30Hz control, 10s requested.
Commands: forward -> turn -> lateral -> stop, smoothstep between bounded targets,
zero requested pitch/roll. Heading and velocity are world-frame commands.

The only environment overrides are episode time limit (to allow 10s), num_envs=1,
and headless camera availability. Gains, limits, initial-state settings and termination
conditions remain upstream. Original episodeLength=300 would time out just before
10 seconds; increasing it is not fall suppression.

Each episode stores T+1 physical states and T applied actions, timestamps, commands,
normalized HLC latent, actual clamped LLC action, PD target angles, DOF force-sensor
readings, foot positions, net-force contact flags, names, quaternion convention,
resolved configs, checkpoint hashes, seed, diagnostics, video and summary.
Force-sensor readings are not asserted to be pure actuator torques, and net-foot-force
contacts are not asserted to be foot-floor-only. Verify those definitions before TT use.
No state is recorded across a reset; the first done terminates collection. Nonfinite
states are saved as failed raw data and must be excluded from reference training.
Exit 0 means continuous duration reached, 2 means episode failed, other errors are
execution failures. Continuity alone does not approve command-tracking accuracy.

Run from this checkout (output path must be new):

```bash
LD_LIBRARY_PATH=/home/cgr-ugrad-2026/miniconda3/envs/rlgpu/lib \
  ../external/a1-calm-main/.venv/bin/python code/motion_tracking/calm/verify_hlc.py \
  --calm-root ../external/a1-calm-main --out runs/calm_reference/hlc_verify_new --seed 0
```

## Gate and deferred work

Both checkpoints load with 351 HLC state/task input, 343 LLC state input,
117 AMP dimensions, 64 latent dimensions and 12 joint actions.
Initial complete execution: 9 steps, 0.30s, terminated without reset; actual velocity
opposes the initial forward command. This is not a valid 10s reference.
Check terminal diagnostics and video; a termination does not by itself prove trunk fall.
No gain/termination relaxation or checkpoint retraining is justified by this result.

Next: confirm original checkpoint training config/asset/control provenance, inspect initial
state and PD targets, and reproduce the failure through upstream keyboard player with
matching commands. If the chain is correct, obtain the correct policy/config combination.
Do not collect/approve 100 clips from repeatedly reset fragments.

After passing continuity AND measured command fidelity: replace the deterministic command
function with seeded, bounded smooth random command segments (reuse existing target fields),
collect raw successes/failures, then rank successful full-duration clips using measured body-frame
velocity, yaw rate, path curvature and acceleration. Use balanced motion families plus
farthest-point selection within each family; refuse to claim 100 distinct clips when the
valid population is smaller. Persist selection and provenance, keep all raw episodes.
Do not synthesize missed displacement or feed failed source clips to PPO.

Only then convert/retarget a few clips, verify FK/time/contact and valid RSI frames,
run a small MJX multi-reference PPO smoke test, and extend to the requested 20/50/100 stages.
Multi-E and reward-agnostic TT retraining remain separate later experiments. The proposed
80:20 normal/failure transition sampler needs a fixed transition-level labeling convention,
including valid prefixes in failed episodes, before implementation.
