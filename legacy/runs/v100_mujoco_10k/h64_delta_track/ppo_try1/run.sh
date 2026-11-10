set -e
cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=../transformer-transformer/.venv/bin/python
O=runs/v100_mujoco_10k/h64_delta_track/ppo_try1
date '+START %F %T'
for T in circle zigzag straight; do
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY -u code/tt_dynamics/benchmark_task_inference.py \
    --ckpt runs/v100_mujoco_10k/h64_delta_track/tt --out $O/$T --tasks $T --methods gradient rl \
    --device cpu --cpu-threads 4 --seed 0 --candidates 4 --eval-seeds 2 --scales 0 1 \
    --rl-updates 150 --rl-rollouts 8 --rl-lr 0.001 --ppo-epochs 4 > $O/$T.log 2>&1 &
done
wait
date '+END %F %T'
