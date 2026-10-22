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

The joint generation stage still uses Mode 1+4 and the first-window E is
still fixed for later windows. Therefore the current 10-second guidance
**does not optimize E against the complete 10-second reward**. Treat it as
a smoke experiment, not proof of task-optimal morphology. This patch
does not implement shared-E long-horizon optimization.

## Scientific guardrails

- Foot-Anchored is a valid rest start **not** a complete multi-gait data set.
- D4 fixed-S0 sensitivity is one diagnostics measure; optimality is assessed
  by physically valid MJX task reward using morphology-appropriate actions.
- Original TT's reward-agnostic diffusion training objective is unchanged.
- Reward-guided TT joint generation must be validated in MJX.
- Task-specific *differences* in morphology are a hypothesis to test, not
  a success criterion to force.
