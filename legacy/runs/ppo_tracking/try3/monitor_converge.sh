# 20 분마다 수렴 판정. 수렴하면 converged.json 을 남기고 끝난다. 학습은 멈추지 않고, 다음 단계도 시작하지 않는다.
cd /home/cgr-ugrad-2026/quad_ws/codesign/legacy
PY=/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv/bin/python
O=$PWD/runs/ppo_tracking/try3
until $PY $O/converge_check.py > $O/converge_last.json; do sleep 1200; done
cp $O/converge_last.json $O/converged.json
echo "CONVERGED (학습 계속 실행 중, 다음 단계는 사용자 결정) $(cat $O/converged.json) $(date '+%F %T')" >> $O/chain.log
