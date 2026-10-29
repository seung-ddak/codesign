# Shared PPO: staged morphology training and full-horizon evaluation

**Scope:** A1-identical topology `00000011`, 30Hz commands, 300Hz MJX physics, reward-agnostic TT rollout collection. This is a **shared reference-conditioned PPO data collector**, not a replacement for TT reward-guided inference. It changes the original per-motion/transition-expert plan; keep the separate expert pipeline available.

## Changes

- `EpisodeDesignReset.step` skips expensive fresh MJX batch initialization on steps where **no environment finished**. If even one environment finishes, reset happens for the batch and is selected only for terminated lanes. The old terminal transition is still recorded before reset.
- `run_shared_ppo.yaml` now **locks** near_long and near_body, training only near_short initially. A morphology **must pass a 10-second evaluation gate before being unlocked**. A long fixed-timestep curriculum is no longer the default.
- The first phase runs 393,216 transitions with 9 scheduled PPO evaluations and logs scalar PPO metrics to `wandb/.../files/training_metrics.jsonl`. KL/value loss are recorded **only if the PPO trainer exposes them**; do not invent missing metrics.
- `eval_shared_ppo.py` evaluates train and held-out designs and clips separately using the **shared E-conditioned policy**, with native MJX execution through 300 control frames, no autoreset. It writes `metrics.json` and `report.html`.
- `shared_ppo_gate.py` evaluates *already unlocked training morphologies*, **never** silently advances on wall-clock steps. The gate produces an explicit pass/fail exit code. Restart the shared PPO from its latest checkpoint with the next mask if it passes.

## Train, evaluate, and unlock

Start at repository root:

~~~bash
cd ~/quad_ws/codesign
git switch experiment/calm-tt-e000-direct-mjx && git pull --ff-only
WANDB_MODE=offline MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ../transformer-transformer/.venv/bin/python code/motion_tracking/train_rl_procedural.py \
  --config-name run_shared_ppo
~~~

**Before running:** local `runs/calm_pilot_tracking/try4/references/` and its `designs.json` must exist, including clips 001/002/004/006/007 for every tested morphology. `config.pkl` and `checkpoints/<step>` live in the printed W&B run directory; they are not committed to GitHub.

Run a 10-second evaluation using the saved **shared PPO** checkpoint. Replace `<CHECKPOINT>` with its real location (e.g., the printed `.../files/checkpoints/393216`):

~~~bash
../transformer-transformer/.venv/bin/python code/motion_tracking/eval_shared_ppo.py \
  --checkpoint '<CHECKPOINT>' --out runs/calm_shared_ppo/stage1_eval
~~~

Default evaluator runs `near_short`, `near_long`, `near_body`, `e003`, `e008` on all five reference clips. It reports every (design, clip) execution, including **10s survival, joint/body/foot/orientation error, foot slip** and the strict screening result. A short rollout is **not** treated as a successful 10-second trial. If e000 reference data is restored, add `--designs e000 near_short near_long near_body e003 e008` to compare to try2.

Check *stage-1* promotion using full 10s metrics on near_short:

~~~bash
../transformer-transformer/.venv/bin/python code/motion_tracking/shared_ppo_gate.py \
  --metrics runs/calm_shared_ppo/stage1_eval/metrics.json --designs near_short \
  --out runs/calm_shared_ppo/stage1_eval/gate.json
~~~

Exit code `0` permits promotion; `2` means **do not** unlock and inspect results. Initial **engineering** thresholds (not research success thresholds): >=4 runs per current design, >=80% 10s survival, survivor mean body RMSE <=0.20m, joint RMSE <=12deg and orientation mean <=10deg.

After promotion, resume from the saved checkpoint, keeping the 13-dimensional morphology observation unchanged:

~~~bash
WANDB_MODE=offline MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ../transformer-transformer/.venv/bin/python code/motion_tracking/train_rl_procedural.py \
  --config-name run_shared_ppo 'design_unlock_steps=[0,0,2147483647]' \
  'restore_ckpt=<CHECKPOINT>'
~~~

Evaluate the new checkpoint using a **new** output path, and run `shared_ppo_gate.py --designs near_short near_long` on its metrics. Only if it passes, resume with `design_unlock_steps=[0,0,0]`. For stage 3, require **all three** training designs to pass. Evaluate `e003`/`e008` and held-out motion `calm_random_007` throughout; **never train on held-out designs** merely to pass the gate.

**Important:** This is an **evaluation-gated, checkpoint-resumed curriculum**, not a single process that changes JAX-compiled constants during an ongoing rollout. Resuming uses the existing policy/value/normalizer; the PPO optimizer state may reinitialize according to the external training implementation. Record that boundary in comparisons.

## Validation and limits

- Smoke tests on the preceding commit ran only 8 control steps. They verify wiring but do **not** certify 10-second tracking, generalization or convergence.
- Run `pytest tests/test_multi_design_sampling.py tests/test_shared_ppo_gate.py` in the laboratory TT Python environment. The new conditional reset path needs a fresh MJX GPU regression run; code review alone is not runtime verification.
- During `training_metrics.jsonl` inspection, compare reward, critic loss, KL (if exported), and tracking separately; lower loss is not equivalent to better tracking.
- `eval_shared_ppo.py` converts native MJX snapshots to MuJoCo data at every control step for contact/slip metrics; this is a **measurement script**, not a high-throughput data collector.
- First-frame fixed-reference evaluation deliberately differs from RSI training. It is required to compare policies under matched starts.
- Reference-bank contents are local and are not supplied by this code change. Do not run a long experiment until the first short evaluation and recorder regression pass.

