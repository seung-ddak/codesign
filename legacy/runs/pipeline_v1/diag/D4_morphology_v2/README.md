# D4 morphology sensitivity results

Checkpoint: `/home/cgr-ugrad-2026/quad_ws/codesign/runs/pipeline_v1/tt/h32_multishape_B`
Starts: 199 total (e000=6, e001=192, e002=1 source starts); initial-contact matched: 13; TT diffusion seeds: 8.

The same unmodified qpos0, qvel0 and target_q are replayed on each morphology. Each window has 32 states and 31 executed 50 Hz controls. Falls remain included; numerical failures are flagged and only non-finite metric entries are omitted.

## Physical-validity selection

| Source morphology | Audited candidates | Valid across all E | Selected | Contact-matched selected |
|---|---:|---:|---:|---:|
| e000 | 768 | 6 | 6 | 5 |
| e001 | 768 | 718 | 192 | 8 |
| e002 | 768 | 1 | 1 | 0 |

The held-out source strata are strongly imbalanced because the same unmodified start must avoid more than 5 mm non-foot floor penetration on all three bodies. The aggregate is therefore dominated by e001-origin windows and is not a balanced three-stratum estimate. Only 13 groups have matching initial contacts in all three MJX models; contact-matched results are separately available in the JSON.

## Pairwise response ΔMAE (all valid groups)

| Pair | dx (m) | dy (m) | yaw (deg) | height (m) | joints (rad) |
|---|---:|---:|---:|---:|---:|
| e001-e000 | 0.09266 | 0.0595 | 14.16 | 0.03319 | 0.1019 |
| e002-e000 | 0.08655 | 0.06863 | 23.47 | 0.02047 | 0.1043 |
| e002-e001 | 0.08907 | 0.05506 | 21.99 | 0.03286 | 0.1082 |

Across all 199 starts, response slopes are near zero for yaw and joint changes (about −0.024 to 0.065); e002-vs-e000 forward displacement has slope 0.70. Height response is partial (slopes 0.16–0.54). Model B responds to some morphology effects, but substantially underpredicts yaw, lateral and joint response; the strongly imbalanced, filtered source sample limits generalization. All three morphology runs had at least 182/199 survive the 31 controls and no numerical failures.

## Figures

- figure1_morphology_changes.png: MJX vs TT morphology-induced response.
- figure2_delta_mae.png: response ΔMAE by source morphology.
- figure3_slope.png: through-origin slopes; ideal = 1.
- figure4_pair_scatter.png: per-start MJX vs TT response.
- figure5_state_trajectories.png: mean state trajectories by source morphology.

See d4_morphology.json for per-source, contact-matched, fall, absolute-error and diffusion-seed details;
see d4_morphology.npz for per-group starts, trajectories and predictions.
