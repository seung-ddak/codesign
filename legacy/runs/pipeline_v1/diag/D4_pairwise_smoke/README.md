# D4 pairwise morphology sensitivity

Frozen checkpoint: `/home/cgr-ugrad-2026/quad_ws/codesign/runs/pipeline_v1/tt/h32_multishape_B/checkpoint.pt`

Each pair was sampled and screened independently. qpos, qvel and target_q are identical within pair; both starts must be valid and their initial four-foot MJX contact masks must match. No additional training or state correction.

| Pair | groups (source A/B) | ΔMAE x (m) | y (m) | yaw (deg) | height (m) | joints (rad) |
|---|---:|---:|---:|---:|---:|---:|
| e000 ↔ e001 | insufficient | — | — | — | — | — |
| e000 ↔ e002 | insufficient | — | — | — | — | — |
| e001 ↔ e002 | insufficient | — | — | — | — | — |

Machine-readable per-start trajectories, starts, actions, contacts, falls and forecasts are in each pair's `data.npz`; detailed absolute errors and sample screening are in `d4_pairwise.json`.
