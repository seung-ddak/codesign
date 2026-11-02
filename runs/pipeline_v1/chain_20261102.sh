#!/bin/bash
# 2026-11-02 사용자 지시 "진행할 수 있는 거 진행": 학습 없는 것부터 순서대로.
#   1 v13 사전 기준 평가 (eval_v12parity)      2 8-45 2×2 를 kp30 으로 재수집 (factorial_init_action_kp30)
#   3 v14 = v12 + 접지 계수 ÷4 (8-43 (1)) 학습+평가   4 v15 = v12 + 접지 정규화 ±0.3 (8-43 (2)) 학습+평가
# 한 단계가 실패해도 다음 단계를 계속한다. 종료 코드 = 실패한 단계 수.
cd /mnt/gstore/home/xiangyue/RL/codesign_cleanup || exit 9
PY=../transformer-transformer/.venv/bin/python
EVAL=runs/pipeline_v1/eval_v12parity/run_eval.py
fail=0
step() { local name=$1; shift; echo "=== $(date '+%F %T') START $name"; "$@"; local rc=$?; echo "=== $(date '+%F %T') END $name rc=$rc"; [ $rc -ne 0 ] && fail=$((fail + 1)); return 0; }
step v13_eval        $PY -u $EVAL runs/pipeline_v1/fast_mode24_v13_group_loss
step kp30_factorial  $PY -u runs/pipeline_v1/factorial_init_action_kp30/run.py
step v14_train       $PY -u runs/pipeline_v1/fast_mode24_v14_coef_contact_down/run_train.py
step v14_eval        $PY -u $EVAL runs/pipeline_v1/fast_mode24_v14_coef_contact_down
step v15_train       $PY -u runs/pipeline_v1/fast_mode24_v15_contact_norm03/run_train.py
step v15_eval        $PY -u $EVAL runs/pipeline_v1/fast_mode24_v15_contact_norm03
echo "=== CHAIN DONE failures=$fail"
exit $fail
