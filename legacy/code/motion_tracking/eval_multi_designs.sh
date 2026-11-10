#!/bin/bash
set -e
run="$1"
read -ra designs <<< "${EVAL_DESIGNS:-e000 e001 e002 e003}"
for design in "${designs[@]}"; do
  if [[ -n ${EVAL_CHECKPOINT:-} ]]; then
    ../transformer-transformer/.venv/bin/python code/motion_tracking/eval_motions.py "$run" --design "$design" --checkpoint "$EVAL_CHECKPOINT"
  else
    ../transformer-transformer/.venv/bin/python code/motion_tracking/eval_motions.py "$run" --design "$design"
  fi
done
../transformer-transformer/.venv/bin/python - "$run" "${designs[@]}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "motion_eval"
rows = []
for design in sys.argv[2:]:
    with (root / design / "motion_eval.csv").open(newline="") as file:
        rows.extend(csv.DictReader(file))
with (root / "summary.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
PY
