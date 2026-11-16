#!/bin/bash
# 겹친 비교 영상 렌더링이 끝나면 기존 보고서 (slip_impratio_all) 를 고쳐 같은 경로에 올리고 기존 댓글을 수정한다.
cd /home/cgr-ugrad-2026/quad_ws/codesign/check/slip_check
while tmux has-session -t "=slip_overlay" 2>/dev/null; do sleep 30; done
../../../transformer-transformer/.venv/bin/python code/report.py slip_impratio_all --update 6816025198 \
  "겹친 비교 영상 (A vs impratio 10/30/100) 은 렌더링 중, 끝나면 따로 올림.=>겹친 비교 영상 (A vs impratio 10/30/100, 형상 5 × 모션 5) 을 보고서에 추가함. 원래 색 = A 학습 설정, 반투명 주황 = 테스트 설정. 모션 드롭다운으로 거르기." \
  > logs/after_overlay.log 2>&1
