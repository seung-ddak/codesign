# 동시 환경 수 1024·2048·4096 별 PPO 학습 속도 (학습 1 개, seed 0). 나머지 설정은 bench.sh 와 같음 (6 단계 설정, 형상 L20, walkBackwardSlow, 8M step).
cd /home/cgr-ugrad-2026/quad_ws/codesign/code/motion_tracking
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
S=/home/cgr-ugrad-2026/quad_ws/codesign/check/seed_parallel; R=/home/cgr-ugrad-2026/quad_ws/codesign/check/short_leg/references/L20
E='[0.5,0.5,0.5,0.7,0.7,0.5,0.5,0.5,0.5,0.2667,0.2667,0.5,0.5]'
export WANDB_MODE=offline XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
echo -e "num_envs\twall_s\ttrain_sps" > $S/results/bench_envs.tsv
for NE in 1024 2048 4096; do
  t0=$(date +%s); L=$S/logs/envs$NE.log
  WANDB_DIR=/tmp/claude-1000/wandb_bench $PY -u train_rl_procedural.py multi_design=false choice=00000011 "uniforms=$E" \
    env.reference_dir=$R eval_env.reference_dir=$R "env.clips=[walkBackwardSlow__r00]" "eval_env.clips=[walkBackwardSlow__r00]" \
    train.num_timesteps=8000000 train.num_envs=$NE train.batch_size=1024 train.learning_rate=1e-3 \
    env.penalty_termination=0 env.reward_clip_positive=true eval_env.penalty_termination=0 eval_env.reward_clip_positive=true \
    train.num_eval_envs=4 train.num_evals=3 render_final=false seed=0 hydra.run.dir=/tmp/claude-1000/bench_hydra/envs$NE > $L 2>&1
  sps=$(grep -oE "training/sps: [0-9.]+" $L | tail -1 | awk '{print $2}')
  echo -e "$NE\t$(( $(date +%s)-t0 ))\t$sps" >> $S/results/bench_envs.tsv
done
echo DONE >> $S/results/bench_envs.tsv
