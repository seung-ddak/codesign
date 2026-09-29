#!/bin/bash
# T7 TT ψ DDPO training + report + telegram. Run from codesign/ inside tmux:
#   bash stage2/code/run_t7.sh <t5_run> <t7_out> [updates]
set -euo pipefail
T5=$1 OUT=$2 UPDATES=${3:-100}
P=../transformer-transformer/.venv/bin/python
$P stage2/code/train_hardware_guidance.py --pi-theta "$T5" --out "$OUT" --updates "$UPDATES"
$P stage2/code/make_stage_report.py t7 --run "$OUT" --title "$(basename "$OUT")"
bash ../scripts/tg_send_file.sh "$OUT/report.html" "$(basename "$OUT") 보고서"
