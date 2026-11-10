# e000 action A/B ablation

Question: with the same robot, starts, model, update count, mode 1 and normalization, does adding same-start Swap/Offset MJX executions improve the predicted effect of changing A?

Source: `../action_pairs/e000_seed0.npz` (192 starts, 131 source episodes). `split_manifest.json` assigns **source episodes** to Train/Validation/Test before selecting Swap donors. `paired.npz` contains the re-executed three variants with `swap_donor_group_id`; `model_a_orig.npz` keeps Orig only and `model_b_all.npz` keeps all three. The episode split is 92/20/19 source episodes, or 132/31/29 starts. A and B share the same Orig trajectories. All three variants survived 32 states at 27 Test starts, which form the final comparison set.

Reproduce from `codesign/` (GPU required for MJX and TT training):

```bash
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
../transformer-transformer/.venv/bin/python code/tt_dynamics/prepare_ablation.py --source runs/pipeline_v1/action_pairs/e000_seed0.npz --out-dir runs/pipeline_v1/action_ablation
../transformer-transformer/.venv/bin/python code/tt_dynamics/collect_action_pairs.py --starts-from runs/pipeline_v1/action_pairs/e000_seed0.npz --manifest runs/pipeline_v1/action_ablation/split_manifest.json --out runs/pipeline_v1/action_ablation/paired.npz
../transformer-transformer/.venv/bin/python code/tt_dynamics/prepare_ablation.py --source runs/pipeline_v1/action_pairs/e000_seed0.npz --out-dir runs/pipeline_v1/action_ablation --paired runs/pipeline_v1/action_ablation/paired.npz
../transformer-transformer/.venv/bin/python code/tt_dynamics/train_tt.py --data runs/pipeline_v1/action_ablation/model_b_all.npz --out runs/pipeline_v1/tt/h32_ablation_B --horizon 32 --steps 20000 --batch 32 --modes 1 --diff-steps 5 --norm minmax --clip 1 --sampler ddim --split-key split
../transformer-transformer/.venv/bin/python code/tt_dynamics/train_tt.py --data runs/pipeline_v1/action_ablation/model_a_orig.npz --out runs/pipeline_v1/tt/h32_ablation_A --horizon 32 --steps 20000 --batch 32 --modes 1 --diff-steps 5 --norm minmax --clip 1 --sampler ddim --split-key split --norm-from runs/pipeline_v1/tt/h32_ablation_B/checkpoint.pt
../transformer-transformer/.venv/bin/python code/tt_dynamics/eval_action_pairs.py --data runs/pipeline_v1/action_ablation/model_b_all.npz --old runs/pipeline_v1/tt/h32_ablation_A --new runs/pipeline_v1/tt/h32_ablation_B --split test --out runs/pipeline_v1/action_ablation/comparison.json --plot runs/pipeline_v1/action_ablation/comparison.png
```

The collector and dataset writer refuse to overwrite existing outputs. To reproduce in place, first choose new output paths. Both models have the same random seed and initial weights. B's normalization is computed from its Train rows only; A reads those exact statistics. Test rows never enter either training run.

The result isolates this fixed-start action coverage more cleanly than the previous broad-TT comparison. B also has three times as many distinct training trajectories, so an improvement cannot be attributed exclusively to counterfactual pairing without a data-volume matched control. No morphology generalization or guided task success is measured here.

## Result (2026-10-20)

Both models finished 20,000 updates. `comparison.json` contains the four individual diffusion seeds and responses for every Test start; `comparison.png` summarizes slopes and A-relative error. The table uses the prediction averaged across four seeds. Lower ΔMAE is better; slope 1 means the predicted response has the MJX response magnitude.

| Response | A ΔMAE | B ΔMAE | A slope | B slope |
|---|---:|---:|---:|---:|
| Swap yaw (deg) | 9.69 | 7.62 | 0.83 | 0.64 |
| Offset yaw (deg) | 18.19 | 12.10 | 0.52 | 0.56 |
| Swap joint angles (rad) | 0.154 | 0.125 | 0.45 | 0.59 |
| Offset joint angles (rad) | 0.162 | 0.088 | 0.31 | 0.77 |
| Offset sideways position (m) | 0.093 | 0.056 | 0.07 | 0.56 |

B lowers both primary error metrics for both interventions; the four individual seeds agree on that direction. On a per-start basis, B improves joint-angle error at 23/27 Swap and 27/27 Offset starts, and yaw error at 13/27 Swap and 18/27 Offset starts. Swap yaw slope gets worse even though its mean absolute error falls, and Swap height error rises slightly (0.0146 → 0.0154 m). This supports a benefit to action-response prediction in this e000 Test set, with uneven gains across states and interventions.

## Three-morphology follow-up (2026-10-20)

Models A and B were retrained on e000/e001/e002 with the same architecture, 20,000 updates, mode 1, and held-out episode splits. A uses Orig only (576 total episodes: 400 train/84 validation/92 test); B uses Orig/Swap/Offset (1,728 total: 1,200/252/276). Evaluation used the corresponding Test split for each trained morphology; groups survive all variants at e000/e001/e002: 27/23/22. Lower ΔMAE is better.

| Morphology (n) | Swap yaw (deg) | Offset yaw (deg) | Swap joints (rad) | Offset joints (rad) | Swap Δx (m) | Offset Δy (m) |
|---|---:|---:|---:|---:|---:|---:|
| e000 (27) | 8.81 → 9.31 | 15.33 → 14.70 | 0.161 → 0.125 | 0.142 → 0.084 | 0.080 → 0.090 | 0.083 → 0.063 |
| e001 (23) | 14.63 → 14.07 | 16.13 → 14.06 | 0.176 → 0.160 | 0.152 → 0.106 | 0.083 → 0.087 | 0.104 → 0.087 |
| e002 (22) | 13.97 → 12.69 | 19.56 → 17.79 | 0.137 → 0.114 | 0.160 → 0.097 | 0.060 → 0.063 | 0.102 → 0.077 |

B lowers joint ΔMAE on all six morphology/intervention combinations, yaw ΔMAE on five of six (e000 Swap is worse), and lateral-position ΔMAE on all three; forward-position ΔMAE is slightly worse on all three. The training sets differ threefold in episode count, so the gains cannot be attributed to pairing alone. This tests held-out episodes only on the three training morphologies, not held-out morphology or task success. Full outputs: `multishape_test_e000/e001/e002.json` and matching PNGs.
