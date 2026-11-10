set -e
cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=$PWD/../transformer-transformer/.venv/bin/python
O=$PWD/runs/ppo_tracking/try1
date '+START retarget %F %T'
python3 -c "import json;print('\n'.join(r['design'] for r in json.load(open('$O/plan.json'))['designs']))" | \
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 xargs -P 14 -I{} sh -c "$PY $O/retarget_one.py {} > $O/logs_retarget_{}.log 2>&1"
test $(ls $O/ready | wc -l) -eq 16
date '+START train %F %T'
cd code/motion_tracking
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -u shared_calm_train.py --out $O --stage 0 --visit 0 --steps 15000000 --seconds 3600 > $O/train.log 2>&1
date '+END %F %T'
