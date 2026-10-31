#!/usr/bin/env bash
set -euo pipefail
cd /home/cgr-ugrad-2026/quad_ws/codesign
while kill -0 200969 2>/dev/null; do sleep 10; done
checkpoint=$(../transformer-transformer/.venv/bin/python - <<'PY'
from pathlib import Path
p=Path('runs/calm_pilot_tracking/try2')
items=[d for d in p.glob('wandb/offline-run-*/files/checkpoints/*') if d.name.isdigit() and (d/'_CHECKPOINT_METADATA').is_file()]
last=max(items,key=lambda d:int(d.name));assert int(last.name)>=5242880,'Full training budget has not completed'
print(last.resolve())
PY
)
export JAX_PLATFORMS=cpu MUJOCO_GL=egl MPLCONFIGDIR=/tmp/mpl-calm
../transformer-transformer/.venv/bin/python -u code/motion_tracking/eval_calm_pd_tracking.py \
 --reference runs/calm_pilot_retarget/try3/references/e000/00000011 \
 --out runs/calm_pilot_tracking/try2/evaluation --checkpoint "$checkpoint"
../transformer-transformer/.venv/bin/python runs/calm_pilot_tracking/try2/build_report.py
