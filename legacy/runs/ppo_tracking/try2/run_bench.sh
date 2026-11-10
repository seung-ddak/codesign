cd /home/cgr-ugrad-2026/quad_ws/codesign/code/motion_tracking
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
O=/home/cgr-ugrad-2026/quad_ws/codesign/runs/ppo_tracking/try2
for N in 32 256 1024; do
  date "+START bench $N %T"
  XLA_PYTHON_CLIENT_PREALLOCATE=false timeout 900 $PY -u shared_calm_train.py --out $O/bench_$N --stage 0 --visit 0 --steps 200000000 --seconds 300 > $O/bench_$N/train.log 2>&1
  echo "exit $? $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
done
date "+END bench %T"
