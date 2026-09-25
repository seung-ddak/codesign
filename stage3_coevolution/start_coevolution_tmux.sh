#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
run_dir=${1:?Usage: start_coevolution_tmux.sh RUN_DIR}
run_dir=$(realpath "$run_dir")
python="$repo/../transformer-transformer/.venv/bin/python"
session="coevolution_$(basename "$run_dir")"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Already running: $session"
  exit 0
fi

mapfile -d '' -t options < <("$python" - "$run_dir/config.json" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1]))
for key, value in config.items():
    if key in ("out", "check", "resume"):
        continue
    print("--" + key.replace("_", "-"), end="\0")
    print(str(value), end="\0")
PY
)
printf -v command '%q ' "$python" -u "$repo/stage3_coevolution/train_coevolution.py" \
  --resume --out "$run_dir" "${options[@]}"
printf -v log '%q' "$run_dir/train.log"
tmux new-session -d -s "$session" -c "$repo" "$command >> $log 2>&1"
echo "Started $session; log: $run_dir/train.log"
