#!/bin/bash
# After a training tmux session ends: rename the run folder, evaluate, record videos, build report.html.
# Run from codesign/ inside tmux (scripts/tmux_run.sh sends the finish notification):
#   bash stage2/code/chain_after_train.sh <train_session> <train_out> <run_dir> "<task>" "<title>" [baseline_run]
# Sends report_tg.html (videos inlined) to telegram; publishing report.html as an artifact is manual (Claude).
set -euo pipefail
SESSION=$1 SRC=$2 RUN=$3 TASK=$4 TITLE=$5 BASE=${6:-}
P=../transformer-transformer/.venv/bin/python

while tmux has-session -t "=$SESSION" 2>/dev/null; do sleep 60; done
# Training must have saved its final checkpoint (brax names checkpoints by step).
want=$($P -c "import json; print(int(float(json.load(open('$SRC/config.json'))['args']['timesteps'])))")
last=$(ls "$SRC/checkpoints" | sort | tail -1)
[ "$((10#$last))" -ge "$want" ] || { echo "training incomplete: last checkpoint $last < $want"; exit 1; }
[ "$SRC" = "$RUN" ] || mv "$SRC" "$RUN"

$P stage2/code/eval_locomotion_policy.py eval --run "$RUN"
mid=$(( $($P -c "import json; print(sum(len(g) for g in json.load(open('$RUN/designs.json'))))") / 2 - 1 ))
$P stage2/code/record_eval_video.py --run "$RUN" --out "$RUN/videos" --ranks 0 "$mid" -1
$P stage2/code/record_eval_video.py --run "$RUN" --out "$RUN/videos" --ranks 0 "$mid" -1 --fixed
$P stage2/code/make_report.py --run "$RUN" --task "$TASK" --title "$TITLE" ${BASE:+--baseline "$BASE"}
bash ../scripts/tg_send_file.sh "$RUN/report_tg.html" "$(basename "$RUN") 보고서"
