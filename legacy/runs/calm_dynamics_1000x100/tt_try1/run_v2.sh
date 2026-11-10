set -e
cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=$PWD/../transformer-transformer/.venv/bin/python
O=$PWD/runs/calm_dynamics_1000x100/tt_try1
date '+START tasks_v2 %F %T'
TT_REWARD_PROFILE=v2 CUDA_VISIBLE_DEVICES="" $PY -u code/tt_dynamics/benchmark_task_inference.py --ckpt $O/tt --out $O/task_v2 \
 --tasks circle zigzag straight --methods gradient rl --device cpu --cpu-threads 12 --seed 0 --candidates 4 --eval-seeds 2 \
 --scales 0 1 3 10 --rl-updates 20 --rl-rollouts 4 --rl-lr 0.001 --ppo-epochs 4 > $O/task_v2.log 2>&1
date '+START videos_v2 %F %T'
source env_osmesa.sh
TT_REWARD_PROFILE=v2 CUDA_VISIBLE_DEVICES="" $PY $O/render_task_pair.py --root $O/task_v2 --tag _v2 > $O/task_video_v2.log 2>&1
date '+END %F %T'
