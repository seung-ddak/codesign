set -e
cd /home/cgr-ugrad-2026/quad_ws/codesign/code/motion_tracking
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
O=/home/cgr-ugrad-2026/quad_ws/codesign/runs/ppo_tracking/try1
test $(ls $O/ready | wc -l) -eq 16
date '+START train %F %T'
XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -u shared_calm_train.py --out $O --stage 0 --visit 0 --steps 15000000 --seconds 3600 > $O/train.log 2>&1
date '+END %F %T'
