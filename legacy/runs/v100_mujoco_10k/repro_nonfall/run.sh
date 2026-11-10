cd /home/cgr-ugrad-2026/quad_ws/codesign
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 ../transformer-transformer/.venv/bin/python -u runs/v100_mujoco_10k/repro_nonfall/repro_check.py --ckpt runs/v100_mujoco_10k/tt/h64_warmup_ema_mode0124/checkpoint.pt --data runs/local_mujoco_10k_20261109/rollouts --out runs/v100_mujoco_10k/repro_nonfall --n 100 > runs/v100_mujoco_10k/repro_nonfall/run.log 2>&1
echo "EXIT $?" >> runs/v100_mujoco_10k/repro_nonfall/run.log
