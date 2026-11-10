cd /home/cgr-ugrad-2026/quad_ws/codesign
O=/home/cgr-ugrad-2026/quad_ws/codesign/runs/v100_mujoco_10k/h64_delta_trim
date '+START %F %T' > $O/train.log
systemd-run --user --scope -q -p MemoryMax=18G -p MemorySwapMax=0 ../transformer-transformer/.venv/bin/python -u code/tt_dynamics/train_tt.py \
 --data $(cat $O/data_list.txt) --out $O/tt --horizon 64 --steps 60000 --batch 64 --lr 0.0002 --d-model 256 --layers 6 \
 --diff-steps 5 --norm minmax --norm-quantile 0.005 --clip 1 --sampler ddim --val-frac .2 --seed 0 --modes 0 1 2 4 \
 --mode-weights .1 .2 .35 .35 --film --eval-episodes 8 --warmup-steps 500 --ema-power .75 --checkpoint-every 2000 \
 --val-every 500 --pose delta >> $O/train.log 2>&1
echo "EXIT $?" >> $O/train.log
date '+END %F %T' >> $O/train.log
