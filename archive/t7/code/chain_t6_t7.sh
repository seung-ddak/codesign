#!/bin/bash
# After a T5 run's post-train chain: T5 gate -> T6 check (+report) -> T6 gate -> T7 (+report).
# Stops (with a one-line telegram) at the first failed gate. Run from codesign/ inside tmux:
#   bash stage2/code/chain_t6_t7.sh <wait_session> <t5_run> <t6_out> <t7_out>
set -uo pipefail
WAIT=$1 T5=$2 T6=$3 T7=$4
P=../transformer-transformer/.venv/bin/python
TG=../scripts/tg_send.sh

while tmux has-session -t "=$WAIT" 2>/dev/null; do sleep 60; done
$P stage2/code/t6_evaluator_check.py t5 --run "$T5" || { bash "$TG" "$(basename "$T5") T5 기준 미달, T6 보류"; exit 3; }
$P stage2/code/t6_evaluator_check.py t6 --run "$T5" --out "$T6"; t6=$?
$P stage2/code/make_stage_report.py t6 --run "$T6" --title "$(basename "$T6")" \
  && bash ../scripts/tg_send_file.sh "$T6/report.html" "$(basename "$T6") 보고서"
[ $t6 -eq 0 ] || { bash "$TG" "$(basename "$T6") T6 기준 미달, T7 보류"; exit 3; }
bash stage2/code/run_t7.sh "$T5" "$T7" 100
