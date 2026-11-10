# D4 pairwise morphology sensitivity

Frozen checkpoint: `/home/cgr-ugrad-2026/quad_ws/codesign/runs/pipeline_v1/tt/h32_multishape_B/checkpoint.pt`

Each pair was sampled and screened independently. qpos, qvel and target_q are identical within pair; both starts must be valid and their initial four-foot MJX contact masks must match. No additional training or state correction.

| Pair | groups (source A/B) | ΔMAE x (m) | y (m) | yaw (deg) | height (m) | joints (rad) |
|---|---:|---:|---:|---:|---:|---:|
| e000 ↔ e001 | 1 (0/1) | 0.03188 | 0.1231 | 2.66 | 0.08928 | 0.2358 |

### e000 ↔ e001

Source counts: {'e000': 0, 'e001': 1}; initial contacts match on all 1 groups. Falls: e000 0/1, e001 0/1.

| Response | MJX RMS Δ | TT RMS Δ | slope | sign agreement | ΔMAE |
|---|---:|---:|---:|---:|---:|
| dx_m | 0.03888 | 0.07076 | 1.82 | 1.000 | 0.03188 |
| dy_m | 0.139 | 0.01593 | 0.1146 | 1.000 | 0.1231 |
| yaw_deg | 1.858 | 4.519 | 2.432 | 1.000 | 2.66 |
| height_m | 0.04499 | 0.04429 | -0.9845 | 0.000 | 0.08928 |
| joint_rad | 0.3637 | 0.07199 | 0.04012 | 0.750 | 0.2358 |

Figures: `e000_e001/response_scatter.png`, `e000_e001/state_trajectories.png`.
| e000 ↔ e002 | 1 (0/1) | 0.1273 | 0.1041 | 0.07098 | 0.03708 | 0.04765 |

### e000 ↔ e002

Source counts: {'e000': 0, 'e002': 1}; initial contacts match on all 1 groups. Falls: e000 0/1, e002 0/1.

| Response | MJX RMS Δ | TT RMS Δ | slope | sign agreement | ΔMAE |
|---|---:|---:|---:|---:|---:|
| dx_m | 0.2355 | 0.1082 | 0.4595 | 1.000 | 0.1273 |
| dy_m | 0.1132 | 0.009095 | 0.08036 | 1.000 | 0.1041 |
| yaw_deg | 3.348 | 3.419 | 1.021 | 1.000 | 0.07098 |
| height_m | 0.05299 | 0.01591 | 0.3002 | 1.000 | 0.03708 |
| joint_rad | 0.07189 | 0.02348 | 0.09513 | 0.833 | 0.04765 |

Figures: `e000_e002/response_scatter.png`, `e000_e002/state_trajectories.png`.
| e001 ↔ e002 | insufficient | — | — | — | — | — |

Machine-readable per-start trajectories, starts, actions, contacts, falls and forecasts are in each pair's `data.npz`; detailed absolute errors and sample screening are in `d4_pairwise.json`.
