# GPU 하나에서 PPO 학습을 N 개 (시드 다름) 동시에 돌릴 때 학습 속도. 6 단계 설정 (동시 환경 1,024, batch 1,024, unroll 24, 50 Hz / 200 Hz, 피라미드·impratio 1·반복 5/7).
# 형상 L20, 모션 walkBackwardSlow, 프로세스마다 8M step. 출력: logs/N<n>_s<seed>.log, results/bench.tsv
cd /home/cgr-ugrad-2026/quad_ws/codesign/code/motion_tracking
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
S=/home/cgr-ugrad-2026/quad_ws/codesign/check/seed_parallel; R=/home/cgr-ugrad-2026/quad_ws/codesign/check/short_leg/references/L20
E='[0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.5,0.5,0.2667,0.2667,0.5,0.5]'
export WANDB_MODE=offline XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
echo -e "N\tseed\twall_s\ttrain_sps" > $S/results/bench.tsv
for N in 1 2 3 4; do
  for ((i=0;i<N;i++)); do
    ( t0=$(date +%s); L=$S/logs/N${N}_s$i.log
      WANDB_DIR=/tmp/claude-1000/wandb_bench $PY -u train_rl_procedural.py multi_design=false choice=00000011 "uniforms=$E" \
        env.reference_dir=$R eval_env.reference_dir=$R "env.clips=[walkBackwardSlow__r00]" "eval_env.clips=[walkBackwardSlow__r00]" \
        train.num_timesteps=8000000 train.num_envs=1024 train.batch_size=1024 train.learning_rate=1e-3 \
        env.penalty_termination=0 env.reward_clip_positive=true eval_env.penalty_termination=0 eval_env.reward_clip_positive=true \
        train.num_eval_envs=4 train.num_evals=3 render_final=false seed=$i hydra.run.dir=/tmp/claude-1000/bench_hydra/N${N}_s$i > $L 2>&1
      sps=$(grep -oE "training/sps: [0-9.]+" $L | tail -1 | awk '{print $2}')
      echo -e "$N\t$i\t$(( $(date +%s)-t0 ))\t$sps" >> $S/results/bench.tsv ) &
  done
  wait
done
echo DONE >> $S/results/bench.tsv
