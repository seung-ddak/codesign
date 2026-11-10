# (사용자가 진행을 정한 뒤 직접 실행) 데이터 수집 → TT (100 단계) → task (reward gradient / RL) → 영상.
#   tmux kill-session -t =ppo_try2_train; tmux new-session -d -s ppo_try2_pipeline "bash runs/ppo_tracking/try2/run_pipeline.sh"
# 이미 끝난 단계는 건너뛴다 (다시 실행해도 됨).
cd /home/cgr-ugrad-2026/quad_ws/codesign
PY=$PWD/../transformer-transformer/.venv/bin/python
O=$PWD/runs/ppo_tracking/try2
log(){ echo "$1 $(date '+%F %T')" >> $O/chain.log; }
# 수렴 판정은 monitor_converge.sh 가 따로 한다. 이 스크립트는 사용자가 진행을 정한 뒤 직접 실행한다.
# 학습이 아직 돌고 있으면 시작하지 않는다 (학습 세션을 먼저 멈출 것).
if tmux has-session -t "=ppo_try2_train" 2>/dev/null; then echo "학습 세션이 아직 실행 중이다. 먼저 멈춘다." >&2; exit 1; fi
set -e
log "START collect"
for D in $(python3 -c "import json;print(' '.join(r['design'] for r in json.load(open('$O/plan.json'))['designs']))"); do
  for V in 0 1; do
    [ -f $O/rollouts/$D/variant_$V.npz ] && continue
    (cd code/motion_tracking && XLA_PYTHON_CLIENT_PREALLOCATE=false $PY -u shared_calm_collect.py --out $O --design $D --variant $V > $O/logs_collect_${D}_$V.log 2>&1)
  done
done
ls $O/rollouts/e*/variant_*.npz | sort | tr '\n' ' ' > $O/data_list.txt
test $(wc -w < $O/data_list.txt) -eq 40
log "START tt_d100"
if [ ! -f $O/tt_d100/checkpoint.pt ]; then
systemd-run --user --scope -q -p MemoryMax=18G -p MemorySwapMax=0 $PY -u code/tt_dynamics/train_tt.py \
 --data $(cat $O/data_list.txt) --out $O/tt_d100 --horizon 64 --steps 60000 --batch 64 --lr 0.0002 --d-model 256 --layers 6 \
 --diff-steps 100 --norm minmax --norm-quantile 0.005 --clip 1 --sampler ddim --val-frac .2 --seed 0 --modes 0 1 2 4 \
 --mode-weights .1 .2 .35 .35 --film --eval-episodes 8 --warmup-steps 500 --ema-power .75 --checkpoint-every 2000 \
 --val-every 500 --pose delta > $O/tt_d100_train.log 2>&1
fi
log "START tasks"
TT_CONTROL_HZ=30 TT_PHYSICS_HZ=240 TT_REWARD_PROFILE=v2 CUDA_VISIBLE_DEVICES="" $PY -u code/tt_dynamics/benchmark_task_inference.py \
 --ckpt $O/tt_d100 --expected-diffusion-steps 100 --out $O/task_d100 --tasks circle zigzag straight --methods gradient rl \
 --device cpu --cpu-threads 12 --seed 0 --candidates 4 --eval-seeds 2 --scales 1 3 --rl-updates 20 --rl-rollouts 4 \
 --rl-lr 0.001 --ppo-epochs 4 --resume > $O/task_d100.log 2>&1
log "START videos"
source env_osmesa.sh
TT_CONTROL_HZ=30 TT_PHYSICS_HZ=240 TT_REWARD_PROFILE=v2 CUDA_VISIBLE_DEVICES="" $PY runs/calm_dynamics_1000x100/tt_try1/render_task_pair.py --root $O/task_d100 > $O/task_video.log 2>&1
log "END pipeline"
