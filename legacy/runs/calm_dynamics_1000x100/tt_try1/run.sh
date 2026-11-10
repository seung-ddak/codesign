set -e
cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=$PWD/../transformer-transformer/.venv/bin/python
O=$PWD/runs/calm_dynamics_1000x100/tt_try1
while tmux has-session -t "=render_e0000" 2>/dev/null; do sleep 10; done
date '+START tt_train %F %T'
systemd-run --user --scope -q -p MemoryMax=18G -p MemorySwapMax=0 $PY -u code/tt_dynamics/train_tt.py \
 --data runs/calm_dynamics_1000x100/rollouts/e0000/primitive_seed0.npz --out $O/tt --horizon 64 --steps 60000 --batch 64 --lr 0.0002 --d-model 256 --layers 6 \
 --diff-steps 5 --norm minmax --norm-quantile 0.005 --clip 1 --sampler ddim --val-frac .2 --seed 0 --modes 0 1 2 4 \
 --mode-weights .1 .2 .35 .35 --film --eval-episodes 8 --warmup-steps 500 --ema-power .75 --checkpoint-every 2000 \
 --val-every 500 --pose delta > $O/tt_train.log 2>&1
date '+START tasks %F %T'
CUDA_VISIBLE_DEVICES="" $PY -u code/tt_dynamics/benchmark_task_inference.py --ckpt $O/tt --out $O/task_v1 \
 --tasks circle zigzag straight --methods gradient rl --device cpu --cpu-threads 4 --seed 0 --candidates 2 --eval-seeds 1 \
 --scales 0 1 --rl-updates 2 --rl-rollouts 4 --rl-lr 0.001 --ppo-epochs 4 > $O/task_v1.log 2>&1
date '+END %F %T'
