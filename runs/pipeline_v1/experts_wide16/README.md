# Wide 16-morphology expert experiment

Training designs: e000, e001, e002, e004–e016 (16 total). e003 remains held out. Unlike the previous midpoint-only experiment, e008, e012, e014, e015, and e016 extend the six-design parameter range in 12 of 13 dimensions; the other five new designs interpolate between these candidates and e000.

All 19 selected motions have retargeted references for each design. The ten new designs have at most 3.2 mm foot tracking error, no joint-limit frames in their retarget summaries, and at most 4.4 mm non-foot floor penetration (5 mm exclusion threshold). Inverse-dynamics bad fractions remain high for some fast motions, so expert evaluation is necessary before rollout collection.

The 15,360-step squat smoke run completed. Full training targets 14,000,000 steps per motion so each morphology receives about the same sample budget as in the six-design run. The first squat checkpoint at 583,680 steps was retained when this budget was increased. Training runs in the `expert_wide16` tmux session. Progress: `queue.log`; outputs: `<motion>/train.log` or `resume.log`, `eval.log`, final checkpoint and 17-design evaluation under the W&B run. Completed results are pushed to the `experiment/wide16-experts` branch by `publish_wide16_experts.sh`.
