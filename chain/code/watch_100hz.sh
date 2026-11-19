#!/bin/bash
# 사용: Claude Code 세션에서 Monitor 로 실행 (30 분마다 재무장). 100Hz 학습 (run_100hz.sh) 알림용.
# 모션 세트별 리타겟/학습 완료·오류를 한 번씩만 출력 (seen 파일로 재무장 시 중복 방지)
C=/home/cgr-ugrad-2026/quad_ws/codesign/chain; S=$C/logs/watch_seen.txt; touch $S
emit(){ grep -qxF "$1" $S || { echo "$1" >> $S; echo "$1"; }; }
while true; do
  for d in $C/runs/s5h_retarget100/ready/*/; do [ -d "$d" ] || continue; g=$(basename $d)
    [ "$(ls $d | wc -l)" -ge 9 ] && emit "리타겟 완료: $g"; done
  for f in $C/runs/s6h_train100/*/done; do [ -f "$f" ] && emit "학습 완료: $(basename $(dirname $f))"; done
  for L in s5h s6h s7h s8z; do
    grep -q "^EXIT" $C/logs/$L.log 2>/dev/null && emit "$L 종료: $(grep '^EXIT' $C/logs/$L.log | tail -1)"
    grep -qE "Traceback|Error|Killed|OOM" $C/logs/$L.log 2>/dev/null && emit "$L 오류: $(grep -E 'Traceback|Error|Killed|OOM' $C/logs/$L.log | tail -1 | cut -c1-120)"
  done
  grep -q "s7h END" $C/logs/chain.log 2>/dev/null && emit "평가 완료: runs/s7h_eval100/report.html"
  grep -q "s8z retarget 완료" $C/logs/chain.log 2>/dev/null && emit "s8z 리타겟 완료 (형상 500)"
  grep -q "s8z END" $C/logs/chain.log 2>/dev/null && emit "rollout 수집 완료: runs/s8z_collect/report.html"
  [ -f $C/logs/s6h.log ] && ! grep -q "^EXIT" $C/logs/s6h.log && { tmux has-session -t =s6h 2>/dev/null || emit "s6h tmux 세션 사라짐"; }
  grep -q "s5z assignment" $C/logs/chain.log && emit "z 수집·배정 완료: $(grep "s5z assignment" $C/logs/chain.log | tail -1 | cut -c21-)"
  sleep 60
done
