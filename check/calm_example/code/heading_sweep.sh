# 회전 명령 변환 방식 후보를 같은 무작위 계획 12 개 (make_plan(12, seed 4242)) 로 수집해 비교. 결과: heading_sweep/<방식>/accepted
D=/home/cgr-ugrad-2026/quad_ws/codesign/check/calm_example; R=/home/cgr-ugrad-2026/miniconda3/envs/rlgpu
cd /home/cgr-ugrad-2026/quad_ws/codesign
/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python -c "
import json,sys;sys.path.insert(0,'chain/code');import calm_joystick as c
json.dump(dict(motions=c.make_plan(12,4242)),open('$D/heading_sweep/plan.json','w'))"
for cfg in "integrate 0.3 1.0 int03" "integrate 0.2 1.0 int02" "integrate 0.5 1.0 int05" "step 0.3 1.0 step10" "step 0.3 0.5 step05"; do
  set -- $cfg; O=$D/heading_sweep/$4; mkdir -p $O; cp $D/heading_sweep/plan.json $O/
  CALM_HEADING=$1 CALM_MAX_ERR=$2 CALM_STEP_PERIOD=$3 OMP_NUM_THREADS=2 MPLCONFIGDIR=/tmp/calm-mpl LD_LIBRARY_PATH=$R/lib \
    $R/bin/python -u chain/code/collect_calm_joystick.py --out $O --batch 6 --max-attempts 12 --rl-games-wheel legacy/runs/menagerie_calm_check/try4/runtime/rl_games-1.1.4-py3-none-any.whl > $D/logs/sweep_$4.log 2>&1
  echo "$4 $(tail -1 $D/logs/sweep_$4.log)"
done
