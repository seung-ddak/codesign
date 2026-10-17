What this change does: The collector runs fixed motion experts and saves physical state/action sequences for TT training. It switches experts within each episode and records falls. Reviewed load: one process, 19 experts, 256 parallel episodes, up to 500 steps each on a 12 GB RTX 4070 SUPER.

## Must fix (fixed before seed 0)

1. **Last-step switches can break the report** (`code/motion_tracking/collect_rollouts.py:174`)
   - **What this is:** The report counts source/destination motion pairs.
   - **Problem:** If all episodes fall early and the last step switches, the old code reads beyond the saved clip array. At the planned final step it records an unknown destination instead.
   - **Fix:** Save the destination clip from the same physics step and use it directly. Applied; early-ending and full-horizon one-step regression checks pass.
   - **If we skip it:** The NPZ may be saved but pair reporting crashes or loses the destination.

2. **Saved targets can differ from applied targets** (`code/motion_tracking/collect_rollouts.py:158`)
   - **What this is:** target_q is the joint target used by the PD controller.
   - **Problem:** The environment clips policy actions to [-10,10], but the old writer scales the unclipped output. For action 20 and scale 0.25 it saves offset+5 instead of offset+2.5.
   - **Fix:** Apply the same action limits when saving target_q. Applied and checked with out-of-range inputs.
   - **If we skip it:** Downstream learning can associate a state change with a target that was never applied.

3. **Numerical failures can be saved as valid dynamics** (`code/motion_tracking/collect_rollouts.py:148`)
   - **What this is:** TrackMjxEnv replaces NaN fields with previous values and records state_is_nan.
   - **Problem:** The old collector only checks fellover, so a repaired numerical failure can remain valid.
   - **Fix:** Abort without saving the batch if an active episode reports a numerical failure. Applied; neither collected batch triggered it.
   - **If we skip it:** Artificially repaired states can be treated as real transitions.

4. **A rerun silently replaces existing data** (`code/motion_tracking/collect_rollouts.py:55`)
   - **What this is:** Each seed has a fixed output filename.
   - **Problem:** Running the same seed/output again overwrites the NPZ and pair CSV.
   - **Fix:** Refuse to start when either output already exists. Applied; original smoke and seed 0 outputs are retained.
   - **If we skip it:** Earlier experiment results can be lost.

5. **Filesystem dates can select the wrong training run** (`code/motion_tracking/collect_rollouts.py:66`)
   - **What this is:** A motion folder can contain both an initial run and a resumed run.
   - **Problem:** Git checkout or copying can change directory modification times, so they do not reliably identify the latest training run.
   - **Fix:** Select by the timestamped offline-run directory name. Applied; the resumed trotBackwardSlow run was selected.
   - **If we skip it:** Collection may use an older checkpoint after transfer to another PC.

## Should fix (fixed)

6. **The writer lacked a regression check** (`code/motion_tracking/check_rollouts.py:1`)
   - **What this is:** The saved arrays become the next training stage's input.
   - **Problem:** Terminal-switch indexing and action conversion had no repeatable check.
   - **Fix:** Added a small script that exercises the actual aggregation/conversion code and validates supplied NPZ arrays. Smoke and seed 0 pass shape, finite-value, mask, clip continuity, and applicable target conversion checks.
   - **If we skip it:** These errors can recur without being detected before dataset construction.

Verdict: Ship. This verdict covers the collector fixes, not expanding this experiment. Seed 0 has 106/256 falls (41.4%), so the collection plan's branch C stops seeds 1–3. No transition pair was excluded.

Not checked: forced GPU numerical-failure injection, independent physics replay of saved actions, TT training, and other morphologies. Fall causes are not established by the one-second association statistic. Expert settings match except clip/reference paths; kick and initial perturbations are zero. Existing frame matching compares reference poses/contacts, not the live physical pose/velocity; its contribution to failures needs a separate controlled experiment.
