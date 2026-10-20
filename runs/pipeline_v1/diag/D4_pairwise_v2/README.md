# D4 pairwise morphology sensitivity

Frozen checkpoint: `/home/cgr-ugrad-2026/quad_ws/codesign/runs/pipeline_v1/tt/h32_multishape_B/checkpoint.pt`

Each pair was sampled and screened independently. qpos, qvel and target_q are identical within pair; both starts must be valid and their initial four-foot MJX contact masks must match. No additional training or state correction.

| Pair | groups (source A/B) | ΔMAE x (m) | y (m) | yaw (deg) | height (m) | joints (rad) |
|---|---:|---:|---:|---:|---:|---:|
| e000 ↔ e001 | 15 (6/9) | 0.06397 | 0.084 | 31.18 | 0.04839 | 0.1778 |
| e000 ↔ e002 | 21 (7/14) | 0.1183 | 0.101 | 19.72 | 0.02834 | 0.1267 |
| e001 ↔ e002 | 18 (18/0) | 0.1665 | 0.06666 | 24.37 | 0.02905 | 0.1527 |

## Pair-level details

### e000 ↔ e001

Source counts: {'e000': 6, 'e001': 9}; initial contacts match on all 15 groups. Falls: e000 0/15, e001 0/15.

| Response | MJX RMS Δ | TT RMS Δ | slope | sign agreement | ΔMAE |
|---|---:|---:|---:|---:|---:|
| dx_m | 0.1055 | 0.05032 | 0.1597 | 0.667 | 0.06397 |
| dy_m | 0.1131 | 0.02955 | 0.02999 | 0.533 | 0.084 |
| yaw_deg | 43.99 | 4.383 | 0.006195 | 0.600 | 31.18 |
| height_m | 0.06497 | 0.04243 | 0.3411 | 0.667 | 0.04839 |
| joint_rad | 0.3167 | 0.06097 | -0.008791 | 0.567 | 0.1778 |

Figures: `e000_e001/response_scatter.png`, `e000_e001/state_trajectories.png`.

### e000 ↔ e002

Source counts: {'e000': 7, 'e002': 14}; initial contacts match on all 21 groups. Falls: e000 0/21, e002 5/21.

| Response | MJX RMS Δ | TT RMS Δ | slope | sign agreement | ΔMAE |
|---|---:|---:|---:|---:|---:|
| dx_m | 0.1418 | 0.1642 | 0.6129 | 0.810 | 0.1183 |
| dy_m | 0.1227 | 0.02582 | -0.04702 | 0.333 | 0.101 |
| yaw_deg | 26.59 | 5.25 | -0.000349 | 0.571 | 19.72 |
| height_m | 0.05256 | 0.0354 | 0.4743 | 0.810 | 0.02834 |
| joint_rad | 0.2179 | 0.03758 | 0.004766 | 0.512 | 0.1267 |

Figures: `e000_e002/response_scatter.png`, `e000_e002/state_trajectories.png`.

### e001 ↔ e002

Source counts: {'e001': 18, 'e002': 0}; initial contacts match on all 18 groups. Falls: e001 0/18, e002 1/18.

| Response | MJX RMS Δ | TT RMS Δ | slope | sign agreement | ΔMAE |
|---|---:|---:|---:|---:|---:|
| dx_m | 0.1654 | 0.1075 | -0.04881 | 0.444 | 0.1665 |
| dy_m | 0.09769 | 0.03463 | 0.1384 | 0.556 | 0.06666 |
| yaw_deg | 30.75 | 7.747 | -0.05681 | 0.500 | 24.37 |
| height_m | 0.04498 | 0.0217 | 0.3094 | 0.611 | 0.02905 |
| joint_rad | 0.211 | 0.06642 | -0.01032 | 0.472 | 0.1527 |

Figures: `e001_e002/response_scatter.png`, `e001_e002/state_trajectories.png`.

**Source imbalance:** e002-origin candidate windows contributed zero starts that passed the e001/e002 physical-validity and matching-contact screen. This is a valid matched-start comparison on 18 e001-origin states, but it is not balanced across both source rollout distributions.

Machine-readable per-start trajectories, starts, actions, contacts, falls and forecasts are in each pair's `data.npz`; detailed absolute errors and sample screening are in `d4_pairwise.json`.
