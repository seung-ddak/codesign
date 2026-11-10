set -e
cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=../transformer-transformer/.venv/bin/python
O=/home/cgr-ugrad-2026/quad_ws/codesign/runs/v100_mujoco_10k/h64_delta_track
date '+START data %F %T'
$PY -u code/tt_dynamics/make_track_data.py /home/cgr-ugrad-2026/quad_ws/codesign/runs/local_mujoco_10k_20261109/rollouts $O/data_track 0.2 0.1
ls $O/data_track/e*/variant_*.npz | sort | tr '\n' ' ' > $O/data_list.txt
test $(wc -w < $O/data_list.txt) -eq 400
date '+START train %F %T'
systemd-run --user --scope -q -p MemoryMax=18G -p MemorySwapMax=0 $PY -u code/tt_dynamics/train_tt.py \
 --data $(cat $O/data_list.txt) --out $O/tt --horizon 64 --steps 60000 --batch 64 --lr 0.0002 --d-model 256 --layers 6 \
 --diff-steps 5 --norm minmax --norm-quantile 0.005 --clip 1 --sampler ddim --val-frac .2 --seed 0 --modes 0 1 2 4 \
 --mode-weights .1 .2 .35 .35 --film --eval-episodes 8 --warmup-steps 500 --ema-power .75 --checkpoint-every 2000 \
 --val-every 500 --pose delta > $O/train.log 2>&1
date '+START repro %F %T'
cd /home/cgr-ugrad-2026/quad_ws/codesign
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 $PY -u runs/v100_mujoco_10k/repro_nonfall/repro_check.py --ckpt $O/tt/checkpoint.pt --data runs/local_mujoco_10k_20261109/rollouts --out $O/repro --n 100 > $O/repro.log 2>&1
date '+START tasks %F %T'
cd /home/cgr-ugrad-2026/quad_ws/codesign
$PY -u code/tt_dynamics/benchmark_task_inference.py --ckpt $O/tt --out $O/task_v1 --tasks circle zigzag straight --methods gradient rl --device cpu --cpu-threads 4 --seed 0 --candidates 2 --eval-seeds 1 --scales 0 1 --rl-updates 2 --rl-rollouts 4 --rl-lr 0.001 --ppo-epochs 4 > $O/task_v1.log 2>&1
date '+END %F %T'
