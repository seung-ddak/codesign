set -e
cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=$PWD/../transformer-transformer/.venv/bin/python
O=$PWD/runs/ppo_tracking/try1
while tmux has-session -t "=ppo_track_try1" 2>/dev/null; do sleep 30; done
grep -q "^END" $O/chain.log
test -f $O/checkpoint.json
date '+START collect %F %T'
for D in $(python3 -c "import json;print(' '.join(r['design'] for r in json.load(open('$O/plan.json'))['designs']))"); do
  for V in 0 1; do
    (cd code/motion_tracking && XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -u shared_calm_collect.py --out $O --design $D --variant $V > $O/logs_collect_${D}_$V.log 2>&1)
  done
done
ls $O/rollouts/e*/variant_*.npz | sort | tr '\n' ' ' > $O/data_list.txt
test $(wc -w < $O/data_list.txt) -eq 32
date '+START tt_train %F %T'
systemd-run --user --scope -q -p MemoryMax=18G -p MemorySwapMax=0 $PY -u code/tt_dynamics/train_tt.py \
 --data $(cat $O/data_list.txt) --out $O/tt --horizon 64 --steps 60000 --batch 64 --lr 0.0002 --d-model 256 --layers 6 \
 --diff-steps 5 --norm minmax --norm-quantile 0.005 --clip 1 --sampler ddim --val-frac .2 --seed 0 --modes 0 1 2 4 \
 --mode-weights .1 .2 .35 .35 --film --eval-episodes 8 --warmup-steps 500 --ema-power .75 --checkpoint-every 2000 \
 --val-every 500 --pose delta > $O/tt_train.log 2>&1
date '+START tasks %F %T'
TT_CONTROL_HZ=30 TT_PHYSICS_HZ=240 CUDA_VISIBLE_DEVICES="" $PY -u code/tt_dynamics/benchmark_task_inference.py --ckpt $O/tt --out $O/task_v1 \
 --tasks circle zigzag straight --methods gradient rl --device cpu --cpu-threads 4 --seed 0 --candidates 2 --eval-seeds 1 \
 --scales 0 1 --rl-updates 2 --rl-rollouts 4 --rl-lr 0.001 --ppo-epochs 4 > $O/task_v1.log 2>&1
date '+END %F %T'
