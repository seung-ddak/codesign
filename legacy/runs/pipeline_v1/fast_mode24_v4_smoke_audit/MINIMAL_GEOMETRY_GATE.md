# Minimal 2-second E/S0 geometry gate (2026-10-22)

## Goal and why

The joint TT sampler already produces finite E/S/A and propagates task-reward gradients, but the 4+4 samples from the previous FAST gate yielded **zero feasible initial states**. Some S0 states penetrated the floor. Other S0 states had positive foot clearance with no contact but were rejected by the unconditional >=2-foot support rule.

This patch tests the smallest physical *interface* improvement before considering any redesign of TT training, guidance, or the 10-second composite task.

## Exactly what changes

- **Optional bounded post-generation projection** (`--geometry-s0`): after joint diffusion completes, build the **generated morphology E** in MuJoCo and measure floor clearance of all four feet and nonfoot contacts. If penetration is >2mm, calculate the minimum **upward** base-only displacement that clears it, add a 1mm margin and apply it only if it is <=4cm. Re-run MuJoCo contact geometry after modifying S0. Too-large corrections are rejected, not clipped to 4cm. E, joint angles, base orientation and velocities are untouched.
- **Optional noncontact flight test** (`--allow-airborne-s0`): a candidate with ZERO CPU and ZERO MJX foot contacts, all four feet at least 5mm above the floor, and no penetration/self-collision can pass the initial support heuristic. **Only 2s MJX stepping** determines whether the resulting dynamics actually work; flight detection is not a guarantee.
- Both changes are **opt-in** and do not affect existing/default smoke invocations.
- Generated TT `pred` and predicted reward are **not overwritten**. Actual MJX results are based on the explicitly recorded executed initial height. All candidates include `preprojection`, `geometry_projection`, `geometry_adjusted`, `executed_body_height_m` and `initial_support_mode`.
- `smoke_check.json` distinguishes `guided_completed_unmodified_MJX_count` vs `guided_completed_projection_assisted_MJX_count`. `strict_raw_TT_to_MJX_verified` remains false if only projected starts finish.

**Important limitation:** this is *not* a differentiable penetration gradient inside denoising, and **does not prove that the raw TT distribution is physically consistent**. It is a bounded feasibility projection to test TT→MJX connectivity. A projection-assisted result cannot be called unmodified E/S/A co-generation success, morphology optimality, or sim2real validation.

## Run once (GPU server)

```bash
cd ~/quad_ws/codesign
git switch experiment/wide16-experts
git pull --ff-only

QUICK_CKPT=runs/pipeline_v1/fast_mode24_v4/tt_joint \
QUICK_ROOT=runs/pipeline_v1/fast_mode24_v4_geometry_gate/try1 \
QUICK_CANDIDATES=4 \
bash code/tt_dynamics/run_minimal_geometry_gate.sh
```

The script first runs CPU-only contract tests, then exactly one 2s smoke batch for scale 0 and scale 1. It stops on failed checks, missing checkpoint, or existing output and never launches a 10s evaluation. Uses the existing Mode 2/4 checkpoint; no PPO or TT retraining.

Read:

- `runs/pipeline_v1/fast_mode24_v4_geometry_gate/try1/smoke_zigzag_joint/summary.json`: generated E/S0 vs corrected/executed S0, initial foot clearance, rejected cause and actual MJX reward (null when not stepped).
- `.../smoke_check.json`: gradient reach, physical gate, genuine unmodified versus assisted MJX completions.

## Success/failure interpretation

- **No eligible start:** STOP; current minimal geometry fix did not achieve 2s execution. Do not increase the correction cap or sample indefinitely without reconsidering the method.
- **Eligible but falls in 2s:** actual dynamics failed despite a plausible initial geometry. Do not mark success merely from passing the static gate.
- **2s succeeds only after projection:** TT→MJX stepping is technically demonstrated *with a bounded external correction*, **not** raw generated E/S0 physical consistency. Report projection size.
- **2s succeeds without projection:** unmodified candidate is directly feasible at this sample; requires reward comparison for any performance claim.

The fixed CP-Design pipeline remains unchanged (rollouts → reward-agnostic joint TT → task-reward-guided E/S/A generation → MJX check). This is only a smoke-test interface option, not a new training stage or a substitute for geometry-aware modeling in a production generator.

## Commit intent

Purpose, Changes, Success criteria, Scope and unexecuted test Status are included in the implementation commits; code being pushed does not mean that 2s MJX has passed.
