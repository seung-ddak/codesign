# Wide16 expert → Orig/Swap/Offset TT comparison

`code/tt_dynamics/run_wide16_action.sh` collects primitive rollouts from the selected checkpoints of 19 wide16 experts on the 16 trained morphologies (e003 stays unseen). The rollouts live in `../wide16_action/rollouts/`. For each morphology it chooses 192 valid starts, executes Orig/Swap/Offset in MJX, uses the same `episode_split.json` across all morphologies, and chooses Swap donors inside each split. It then trains a Mode-1 TT on Orig only and another on all three trajectories with the same seed, 20,000 updates, and shared train-only normalization.

Motivation: rechecking `../diag/D4_aligned/h32_cross_C/d4_morphology.npz` with **pairwise** initial-contact matching still gives yaw slopes 0.018 (e001/e000, 178 starts), 0.020 (e002/e000, 212), and −0.040 (e002/e001, 56). This makes initial contact mismatch alone an insufficient explanation for the old D4 failure. The pairwise mask compares contacts only for the two morphologies in each response.

The new e000/e001/e002 primitive rollouts use the same episode indices and sampled time indices, but their expert actions differ: the Orig PD targets have e000/e001 RMS difference 0.135 rad and e000/e002 difference 0.109 rad. Thus these same-shape datasets expand morphology coverage without isolating the effect of E at fixed A. D4's MJX replay remains necessary.

The common split assigns all 304 source episode indices once (213 Train, 46 Validation, 45 Test); each morphology's selected starts inherit those labels. This prevents the same episode index from entering Train for one morphology and Test for another when the rollout seeds align.

`<design>/comparison.json` tests D5 on that design's held-out episodes. `d4_e000_e001_e002_{orig,all}/` and `d4_e000_e008_e016_{orig,all}/` test D4 on both models with the same command across morphologies, morphology-specific body-height alignment, and each morphology's actual initial contact. The D4 JSON reports all starts, three-way contact-matched starts, and pairwise contact-matched starts; check their counts and survival before interpreting slopes. The models receive no cross-morphology paired-action training in this run; an improvement in D5 does not establish an improvement in D4.

Run from `codesign/`: `bash code/tt_dynamics/run_wide16_action.sh`. Existing NPZ and completed TT checkpoints are retained on rerun; an incomplete TT directory requires inspection before rerunning. Logs live beside the corresponding output. The experiment requires GPU access.
