#!/bin/bash
cd /home/cgr-ugrad-2026/quad_ws/codesign
while tmux has-session -t "=pushc" 2>/dev/null; do sleep 30; done
cd chain/code && /home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python -u s10_generate.py || exit 1
cd ../.. && chain/code/push_chunks.sh
