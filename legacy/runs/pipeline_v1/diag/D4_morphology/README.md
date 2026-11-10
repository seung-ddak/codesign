# D4 morphology sensitivity results

Checkpoint: `/home/cgr-ugrad-2026/quad_ws/codesign/runs/pipeline_v1/tt/h32_multishape_B`
Starts: 98 total (e000=6, e001=92, e002=0 source starts); initial-contact matched: 13; TT diffusion seeds: 8.

The same unmodified qpos0, qvel0 and target_q are replayed on each morphology. Each window has 32 states and 31 executed 50 Hz controls. Falls remain included; numerical failures are flagged and only non-finite metric entries are omitted.

## Pairwise response ΔMAE (all valid groups)

| Pair | dx (m) | dy (m) | yaw (deg) | height (m) | joints (rad) |
|---|---:|---:|---:|---:|---:|
| e001-e000 | 0.1214 | 0.07393 | 12.99 | 0.03812 | 0.1105 |
| e002-e000 | 0.08399 | 0.06915 | 17.7 | 0.02311 | 0.1003 |
| e002-e001 | 0.1101 | 0.05597 | 17.17 | 0.03567 | 0.1094 |

## Figures

- figure1_morphology_changes.png: MJX vs TT morphology-induced response.
- figure2_delta_mae.png: response ΔMAE by source morphology.
- figure3_slope.png: through-origin slopes; ideal = 1.
- figure4_pair_scatter.png: per-start MJX vs TT response.
- figure5_state_trajectories.png: mean state trajectories by source morphology.

See d4_morphology.json for per-source, contact-matched, fall, absolute-error and diffusion-seed details;
see d4_morphology.npz for per-group starts, trajectories and predictions.
