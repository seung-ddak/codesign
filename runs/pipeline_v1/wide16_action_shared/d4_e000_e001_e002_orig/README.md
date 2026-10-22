# D4 morphology sensitivity results

Checkpoint: `runs/pipeline_v1/wide16_action_shared/tt_orig`
Starts: 576 total (e000=192, e001=192, e002=192 source starts); initial-contact matched: 2; TT diffusion seeds: 8.

The same qpos0 (body z shifted per E to keep the source lowest-foot clearance), qvel0 and target_q are replayed on each morphology. Each window has 32 states and 31 executed 50 Hz controls. Falls remain included; numerical failures are flagged and only non-finite metric entries are omitted.

## Physical-validity selection

| Source morphology | Audited candidates | Valid across all E | Selected | Contact-matched selected |
|---|---:|---:|---:|---:|
| e000 | 555 | 555 | 192 | 1 |
| e001 | 555 | 555 | 192 | 0 |
| e002 | 555 | 555 | 192 | 1 |

The held-out source strata are strongly imbalanced because the same unmodified start must avoid >5 mm non-foot floor penetration on all three bodies. The aggregate is therefore dominated by e001-origin windows and is not a balanced three-stratum estimate. Only 13 groups have matching initial contacts in all three MJX models; those are reported separately in JSON.

## Pairwise response ΔMAE (all valid groups)

| Pair | dx (m) | dy (m) | yaw (deg) | height (m) | joints (rad) |
|---|---:|---:|---:|---:|---:|
| e001-e000 | 0.1025 | 0.06217 | 8.295 | 0.02491 | 0.09491 |
| e002-e000 | 0.1112 | 0.05032 | 8.177 | 0.02556 | 0.09563 |
| e002-e001 | 0.1956 | 0.09532 | 10.53 | 0.03546 | 0.1356 |

## Figures

- figure1_morphology_changes.png: MJX vs TT morphology-induced response.
- figure2_delta_mae.png: response ΔMAE by source morphology.
- figure3_slope.png: through-origin slopes; ideal = 1.
- figure4_pair_scatter.png: per-start MJX vs TT response.
- figure5_state_trajectories.png: mean state trajectories by source morphology.

See d4_morphology.json for per-source, contact-matched, fall, absolute-error and diffusion-seed details;
see d4_morphology.npz for per-group starts, trajectories and predictions.
