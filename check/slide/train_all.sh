# 형상 E0~E4 × 대표 모션 5 개, 형상·모션마다 PPO 정책 하나 (v7 전문가 설정: 동시 환경 32, 5.24M step, lr 1e-3, 미끄러짐 감점 없음).
# 2026-11-10 사용자 결정: 넘어짐·큰 오차는 감점 없이 조기 종료만 (penalty_termination 0), step 보상은 0 이상으로 자름 (DeepMimic 방식).
# 형상 5 개를 동시에 돌리고, 형상 안에서는 모션을 차례로 학습한다. 끝난 정책은 건너뛴다.
cd /home/cgr-ugrad-2026/quad_ws/codesign/code/motion_tracking
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
S=/home/cgr-ugrad-2026/quad_ws/codesign/check/slide
export WANDB_MODE=offline XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
run_shape(){
  K=$1; E=$($PY -c "import sys;sys.path.insert(0,'$S');from shapes import SHAPES;import json;print(json.dumps(SHAPES['$K']['E'],separators=(',',':')))")
  for M in $($PY -c "import sys;sys.path.insert(0,'$S');from shapes import MOTIONS;print(' '.join(MOTIONS))"); do
    R=$S/runs/$K/$M; [ -f $R/done ] && continue; mkdir -p $R
    echo "START $K $M $(date '+%F %T')" >> $S/train.log
    WANDB_DIR=$R $PY -u train_rl_procedural.py multi_design=false choice=00000011 "uniforms=$E" \
      env.reference_dir=$S/references/$K eval_env.reference_dir=$S/references/$K "env.clips=[$M]" "eval_env.clips=[$M]" \
      train.num_timesteps=5242880 train.num_envs=32 train.batch_size=32 train.learning_rate=1e-3 \
      env.penalty_termination=0 env.reward_clip_positive=true eval_env.penalty_termination=0 eval_env.reward_clip_positive=true \
      train.num_eval_envs=4 train.num_evals=2 render_final=false seed=0 hydra.run.dir=$R/hydra > $R/train.log 2>&1 \
      && touch $R/done && echo "END $K $M $(date '+%F %T')" >> $S/train.log || echo "FAIL $K $M $(date '+%F %T')" >> $S/train.log
  done
}
for K in E0 E1 E2 E3 E4; do run_shape $K & done
wait
echo "ALL DONE $(date '+%F %T')" >> $S/train.log
