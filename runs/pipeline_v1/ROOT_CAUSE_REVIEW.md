# Pipeline v1: causal review and first repair

Reviewed remote main `532b5541e`, `report.html`, training masks, task generation, reward and saved diagnostics. Original data and reports are retained. CGR `h32_try4_cgr` completed: window yaw 5.95°, position 8.38 cm; long-rollout position 58.93 cm. This agrees with a continuing dynamics limitation, not a solved task.

## Finding that must be repaired before interpreting try2

`train_tt.py` mode 3 was documented as free joint generation of E/S/A, including s0. Its actual mask was `ke = (mode >= 1)`: E was known for modes 1, 2 **and 3**. Thus free-mode batches received ground-truth E and had zero loss on the E output. `generate_task.py --free-s0` instead hid both E and s0. The checkpoint was asked to perform a condition absent from training.

Reproduction using the actual old function: number of known E dimensions by mode = `[0, 13, 13, 13]`; intended = `[0, 13, 13, 0]`. This does not mean E was never trained: mode 0 trained E with s0 known. The missing condition was simultaneous unknown E and s0.

Fix: one shared `condition_masks` implementation for training, evaluation and generation; mode 3 hides both E and s0. Checkpoints record `conditioning_version=2`; free generation rejects legacy checkpoints or checkpoints never trained with mode 3. Modes 0/1/2 retain their exact masks. Regression covers all four modes, padding, E-loss gradients, matching inference masks and old-checkpoint rejection.

Affected interpretation: try2 shape selection, initial-state generation and its claimed improvement in prediction/physics agreement must be retested. The bug may contribute to these symptoms, but fixing it does **not** prove stable starts, task-sensitive morphology or reliable 10-second dynamics.

## What links the remaining problems

| Bottleneck | Connected symptoms | Evidence / limits |
|---|---|---|
| Shape-conditioned dynamics and coverage are inadequate | Shape response, unseen-shape validity, reward exploiting predicted motion, drift | Only 3 distinct training shapes; weak velocity/turn prediction. These are separate limitations, not one proved cause. |
| Shape is optimized using only the first window | Later task requirements cannot revise E | `generate` sets E once; later windows set `ke=1`. H=32 means 31 intervals = 0.62 s of reward when E is free, not the whole 10 s. |
| Initial-state distribution is not a launch-state distribution | Early falls, generated low/tilted starts | Free mode uses random windows, including pre-fall states. Repairing masks does not change this sampling. Episode start is also not necessarily stationary. |
| Predicted-state execution differs and drifts | Turn overprediction, open-loop falls | Saved sequence diagnostics compare increasingly different physical states; they do not isolate one-step/model error at identical starts. |

There is no evidence that one edit resolves most of P1–P17. The concrete conditioning mismatch is the first blocking correctness issue. For the research objective, shape coverage and reliable shape-conditioned dynamics remain necessary; global task reward must eventually influence E beyond the first window. Increasing guidance alone cannot establish those properties.

## Controlled next experiment

`h32_freefix_cgr`: same published try5 data, 32-step horizon, 100k steps, 4 modes, diffusion 5, minmax, DDIM, seed 0. Only the conditioning bug is changed in learning. Training and standard dynamics/window evaluation run automatically; results publish to `fix/free-conditioning`. Do not simultaneously change fall-data selection, rewards, shape count or horizon.

Acceptance so far: mask regression and real GPU smoke. Full training/evaluation results remain pending until published. After retraining, free generation must be compared with stored try2 using the same tasks, scales and seeds; assess initial-state validity, actual survival, task reward and per-candidate E changes. A lower training loss alone is not success.

## Corrections to strong claims in the report

- Matching comparisons at one seed do not prove matching is irrelevant; observed changes were small in those runs.
- A better short-horizon MLP supports local predictability; it does not prove adequate data for unseen shapes or generated actions.
- Same noise seeds across guidance scales are a useful paired comparison, not inherently a confound. More independent seed groups are still needed.
- A 32-state window spans 31 action intervals (0.62 s at 50 Hz).
- Per-window differences after an open-loop rollout do not remove the different starting-state history; matched-start replay is needed to isolate local dynamics error.
