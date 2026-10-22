# CP-Design generalization patch (code only)

This branch preserves `experiment/wide16-experts` and previous checkpoints.
No MJX rollout, model training, or 10-second guidance was executed here.

## What changed

- `task_reward.py`: introduced **StraightReward**. The zigzag corridor now
  defaults to 0.15 m half-width instead of 0.30 m in `generate_task.py`.
  With amplitude 0.30 m, a constant y=0 trajectory is no longer free of lane
  penalty. Circle keeps its 0.30 m default. Explicit `--half-width` still wins.
- `run_foot_anchored_pipeline.sh`: `tt_joint` includes the existing
  morphology-intervention `morph_*.npz` data. Optionally also reads
  `$FOOT_ROOT/general_data/*.npz`; all source data must have compatible E,
  qpos/qvel/contact/target_q/valid and split metadata.
- `prepare_general_rollouts.py`: convert existing primitive/transition
  MJX rollout NPZ files to additional joint-training input. This does not
  generate new dynamics or guarantee that an expert can act from a new S0.
- `dynamics_gate.json`: the D4 pipeline records diagnostic results but
  **does not auto-approve** joint training solely because contact-matched
  starts exist. Prior `passed: true` semantics were misleading.

## Preparing optional broader motion data

From the codesign repository directory (paths are examples, use actual files):

```bash
PY=../transformer-transformer/.venv/bin/python
$PY code/tt_dynamics/prepare_general_rollouts.py \
  --sources runs/pipeline_v1/wide16_action/rollouts/e000/primitive_seed0.npz \
            runs/pipeline_v1/wide16_action/rollouts/e001/primitive_seed0.npz \
  --out runs/pipeline_v1/foot_anchored_v2/general_data
```

Only add other valid, physically simulated expert/transition NPZ files after
checking each file's morphology E, action convention, actuator timing, contact
definition and state dimensions. Each input file must have 3+ whole episodes.
The script assigns train/val/test at the episode level. **Important:** do not
reuse an episode already present in the action/foot-anchored training set as a
held-out validation/test episode. Distinct files with the same underlying
episodes must be deduplicated before training. For strict held-out-morphology
generalization, split **by E family**, not only by episode; this helper does
not implement held-out-morphology splitting.

The current `FOOT_ROOT` defaults to the historical v1 path. Set it
explicitly to `runs/pipeline_v1/foot_anchored_v2`.

## Dynamics gate / experimental interpretation

The `dynamics` stage writes `passed: false` even when D4 is evaluable.
This is an intentional **manual review gate**, not an automatic performance
score. Review action D5, morphology D4, sample counts, survival, absolute
state error, and predicted-vs-MJX task reward ranking before setting a
decision in a **new reviewed report**. Do not declare success from the
existence of contact-matched D4 pairs. A training-capable supervisor may
approve the gate after reviewing results; do not bypass to produce claims
without actual evaluation.

The original `joint`/`guidance` stages still retain the historical
first-window-only morphology approach. The **new `shared` stage** uses
Mode 5 (fixed E + partially known rest S0) for the first window and Mode 2
(fixed E + known subsequent S0) for later windows. Its shared morphology is
optimized by **SPSA finite-difference estimates from the complete 10-second
TT-predicted task reward**. This is a memory-bounded alternative to exact
end-to-end diffusion gradient backpropagation; do **not** describe it as
analytical 10-second gradient backpropagation or MJX reward optimization.

The algorithm proposes candidate changes to the same E shared by all
windows, regenerates the entire action/state sequence with common diffusion
noise, and accepts a change only when **full predicted** reward rises.
Scale 0 is the unoptimized baseline. Only final optimized candidates are replayed in MJX by default. With
`--shared-e-eval-baseline` (enabled in the `shared` pipeline stage), the
same-scale **unoptimized** morphology and its independently generated
task-specific action are also replayed in MJX, using the same TT noise.
Look for `shared_e_control` and `shared_e_history` in `summary.json`.
Reward improvements can be illusory if the learned dynamics are inaccurate;
compare the two MJX rewards, survival, predicted-vs-MJX reward and initial
physical validity.
No claim of globally optimal morphology is justified.

## Shared-E experiment (after prepare/dynamics; no GPU run done here)

`shared` uses a **new** checkpoint `tt_joint_shared`, trained with
`--modes 1 2 4 5`. Old Mode-4 checkpoints do not support fixed-E rest
conditioning and must not be reused. Initial morphology is sampled from
Mode 4, with inert E coordinates [0, 11, 12] fixed at 0.5. Search stays
in [0.05, 0.95] to avoid extrapolating outside the v2 morphology prior.

For a controlled exploratory experiment, when diagnostics have not been
approved yet:

```bash
FOOT_ROOT=runs/pipeline_v1/foot_anchored_v2 \
FOOT_EXPLORATORY_SHARED=1 \
FOOT_SHARED_CANDIDATES=4 FOOT_SHARED_ITERS=4 \
bash code/tt_dynamics/run_foot_anchored_pipeline.sh shared
```

This flag authorizes **exploration only**: it does not mark Dynamics or
joint generation as validated. Without it, a reviewed dynamics gate is
required. The output directory is
`runs/pipeline_v1/foot_anchored_v2/composite_10s_shared/`.

The new inference flag is `--rest-s0 --shared-e-guidance`. Important
flags: `--shared-e-iters`, `--shared-e-perturb`,
`--shared-e-step-size`, `--expert-config`. Because SPSA evaluates
multiple full-horizon TT rollouts, start with `--candidates 1` and
`--shared-e-iters 1` when checking V100 memory and runtime.

Tests to run on your machine before training:

```bash
../transformer-transformer/.venv/bin/python -m pytest \
  tests/test_cp_task_reward.py tests/test_shared_e_guidance.py
```

The implementation records initial S0's joint limits, collision penetration,
foot height, quaternion norm and MJX foot-ground contact. Physically invalid
generated starts do not establish successful co-design and should be filtered
from reported task performance.

## Research direction: stop treating D4 as the final objective

Keep the existing CP-Design pipeline and separate evidence into (A)
broad multi-morphology primitive/transition dynamics, (B) physical
joint-generation feasibility and short-horizon accuracy, (C) **end-to-end
task reward in MJX**. D4/D5 remain diagnostics. Evaluate straight, zigzag
and circle with the **same fixed TT checkpoint and comparable candidate
budgets**, compare unoptimized and shared-E approaches, and inspect reward
ranking and survival. Do not reward morphology difference itself: different
E per task is a testable hypothesis, not a required result.

A full task-specific morphology benefit claim additionally requires
cross-evaluation where each morphology receives task-appropriate newly
generated actions, rather than merely replaying another task's controls.

## Scientific guardrails

- Foot-Anchored is a valid rest start **not** a complete multi-gait data set.
- D4 fixed-S0 sensitivity is one diagnostics measure; optimality is assessed
  by physically valid MJX task reward using morphology-appropriate actions.
- Original TT's reward-agnostic diffusion training objective is unchanged.
- Reward-guided TT joint generation must be validated in MJX.
- Task-specific *differences* in morphology are a hypothesis to test, not
  a success criterion to force.
