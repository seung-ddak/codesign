#!/bin/bash
# 실험 폴더 정리: 코드 → code/, *.log → logs/ (2026-11-12 사용자 요청). 사용: tidy_folder.sh <폴더> [기다릴 tmux 세션]
set -e
D=$(realpath "$1"); W=$2
if [ -n "$W" ]; then while tmux has-session -t "=$W" 2>/dev/null; do sleep 60; done; fi
cd "$D"
python3 /home/cgr-ugrad-2026/quad_ws/codesign/chain/code/move_code.py .
mkdir -p logs; mv *.log logs/ 2>/dev/null || true
sed -i -E 's#\$S/([a-z_]+\.log)#$S/logs/\1#g; s#L=\$S/after.log#L=$S/logs/after.log#' code/*.sh
echo "TIDY DONE $(date '+%F %T')" >> logs/tidy.log
