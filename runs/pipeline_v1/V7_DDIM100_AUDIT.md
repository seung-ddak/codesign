# V7-DDIM100: audited E/S/A TT co-design experiment

This experimental runner is based on `experiment/wide16-experts` and **does not overwrite or retroactively improve V6/V7 results**. The branch is proposed for review. No school GPU job has been started by this PR.

## Hypothesis and scope

V6 improved short-window fidelity by training 60k optimizer steps with a 32-state horizon, Mode 2/4, and min-max normalization. V7 proposes morphology E-swap to teach the TT that identical joint-target action sequences can produce different MJX outcomes for different robot embodiments. DDIM100 **changes the diffusion schedule from 5 to 100 levels** but does not create missing primitive skills. We have **no prior evidence** that DDIM100 or E-swap will improve actual composite-task performance.

- Preserve A1 knee topology, 16 training E's (e003 strictly excluded), 13 continuous dimensions.
- Preserve reward-agnostic TT pretraining and task reward **only at inference**; no PPO expert retraining, morphology-specific finetuning or closed-loop replanning.
- Reuse original 16-shape primitive rollouts and E-swap intervention samples.
- Train a **new checkpoint from scratch**, with a 100-level cosine diffusion schedule. The training loop samples a single random noise level per batch item; the DDIM sampler executes 100 denoising steps at inference. This is distinct from 60,000 weight updates and 100 physical E-swap states.
- Use generated E and target joint angles A in actual MJX simulation; predicted S and reward **never qualify as physical success**.

## Audited execution

```bash
git fetch origin
git switch experiment/v7-ddim100-audited
git pull --ff-only

# Run on a machine with the existing transformer-transformer environment,
# multi-morph NPZ files, MJX and GPU. Not runnable via GitHub alone.
bash runs/pipeline_v1/run_v7_ddim100.sh prepare
bash runs/pipeline_v1/run_v7_ddim100.sh train
bash runs/pipeline_v1/run_v7_ddim100.sh smoke
bash runs/pipeline_v1/run_v7_ddim100.sh codesign
bash runs/pipeline_v1/run_v7_ddim100.sh full
bash runs/pipeline_v1/run_v7_ddim100.sh report
```

Or run `bash runs/pipeline_v1/run_v7_ddim100.sh all` after confirming GPU availability and available disk space. Defaults include GPU 0; set `CUDA_VISIBLE_DEVICES` and `V7_100_PYTHON` as necessary.

**Expected duration can be significantly longer than DDIM5.** In joint full-episode guidance, the 100 denoising levels each differentiate task reward over a sequence of TT windows; the overall work is much more than simply evaluating a single 100-step denoising trajectory. We deliberately default 10-second evaluations to one candidate per task. Expand only after a successful run.

All outputs are under `runs/pipeline_v1/v7_ddim100` (override with `V7_100_ROOT`). Original V6 and V7 outputs are not overwritten. The runner refuses existing partial results instead of wiping checkpoints; completed stages are retained. It validates the checkpoint's diffusion schedule and Mode 4 conditioning version.

## Decision criteria

1. 2-second scale 0 MJX replay: physically valid candidates and median TT–MJX path error < 0.15 m.
2. Zigzag **and** circle, 4 seconds, 16 fixed morphologies with four seeds per E (plus 16 joint E/A candidates): TT predicted reward must have a significant shape effect (permutation p < 0.05) and its shape-mean Spearman rank correlation with actual MJX reward must exceed 0.6. Also report actual MJX reward, completion, progress, and whether joint-generation reward beats fixed-shape median.
3. 10-second *open-loop* actual MJX replay, scale 0/1/3, one candidate/task by default. Report falls, valid starts, MJX vs predicted reward, and path error; report actual composite-task success separately. No claim of success solely from a positive predicted reward.

The evaluator is `code/tt_dynamics/eval_v7_ddim100.py`. It creates `audit_smoke.json`, `audit_codesign.json`, `audit_full.json`, `audit_all.json` as appropriate. Missing results are `not_run`; no automatic promotion to a successful experiment. Fail-closed CPU tests are in `tests/test_v7_ddim100_audit.py`.

## Important audit findings / limitations

- `generate_task.py` does not presently include `fixed_e` metadata in the JSON summary. This evaluator reconstructs the 16 groups from contiguous repeated E vectors and checks equal seeds per E instead of silently assuming missing fields exist.
- E-swap replays the same A in morphology-dependent, settled home S0 = g(E); these are physically plausible starts, **not** strict same-S0 counterfactuals. Validity masks exclude states after falls. Increased E-swap sample weight might degrade nominal gaits and does not itself learn zigzag motion.
- E-swap data files have to exist locally on the training host; the Git repository does not contain the complete full 8,192-run generated dataset. Its collection, training, and all GPU scores are **pending**, not verified by this commit.
- Existing code and full regression suite are not verified against the remote school GPU. This change adds a runner and evaluation without silently altering existing trained-model implementation.
- A true `--diff-steps 100` comparison must retrain. Loading a previous 5-step checkpoint for generation cannot provide the same learned denoising process.
- The pipeline preserves the current open-loop 10s evaluation even though historic long-horizon error accumulation remains unresolved. Any change to closed-loop, PPO or reward-conditioned TT pretraining requires a new decision.
- Task scores are currently generated on training-relevant geometric patterns (zigzag/circle) and are not evidence of generalization to every unseen composite task.

## Confirmed vs not confirmed

Confirmed in code: baseline settings, new experiment runner, 100-step training argument, checkpoint guards, E-swap input path, fixed/learned E evaluation requests, post-hoc real-MJX-oriented statistical analysis, and offline regression tests.

**Not confirmed:** 100-step convergence, V7 E-swap improvements, actual GPU execution, successful zigzag/circle task completion, accurate joint morphology ranking, or long-horizon accuracy.

## Local corrections (2026-10-26)

- Rejected fixed-E samples serialize as JSON null; valid counts are recorded per morphology. A missing sample still prevents using that shape mean for ranking.
- Reuse requires an exact generation command and checkpoint SHA-256 match in request.json. Older results without this metadata are preserved and rejected; choose a new V7_100_ROOT.
- Checkpoint reuse also checks seed and training input paths. Reusing an E-swap dataset requires matching its stored collection seed.
- Smoke writes its JSON and HTML, then exits unsuccessfully when the dynamics gate fails or is missing. The automatic all chain stops before codesign/full. Individual stages remain available for explicit diagnosis.
- Joint co-design generation records original-resolution videos. Smoke, fixed-E and 10s full runs remain numerical evaluations. report.html shows predicted-reward and actual-MJX-reward joint winners side by side; other joint videos are collapsed.
- The report stage only reads existing results and videos. It does not rerun simulation. GPU execution of these corrections has not been performed.
