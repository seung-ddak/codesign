set -e
cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=$PWD/../transformer-transformer/.venv/bin/python
O=$PWD/runs/ppo_tracking/try2
date '+START retarget %F %T'
python3 -c "import json;print('\n'.join(r['design'] for r in json.load(open('$O/plan.json'))['designs']))" | \
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 xargs -P 14 -I{} sh -c "$PY $O/retarget_one.py {} > $O/logs_retarget_{}.log 2>&1"
test $(ls $O/ready | wc -l) -eq 20
date '+END retarget %F %T'
