#!/bin/bash
# TT 학습(tmux s9_200k) 끝난 뒤 영상·보고서를 폴더 단위로 나눠 커밋·푸시
cd /home/cgr-ugrad-2026/quad_ws/codesign
while tmux has-session -t "=s9_200k" 2>/dev/null; do sleep 60; done
git add .gitignore chain/code && git commit -qm "s9 보고서 스크립트, 영상 추적 설정

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin experiment/h64-delta-trim
for d in chain/runs/s1_shape_graph chain/runs/s3_generalization chain/runs/s6c_calm chain/runs/s6c50_calm chain/runs/s7_eval chain/runs/s8_collect chain/runs/s9_tt chain/runs/s9_tt_200k check/calm_example check/short_leg check/slide check/slip_check/videos check/slip_check/videos_overlay check/try3_contact; do
  git add "$d" && git diff --cached --quiet && continue
  git commit -qm "영상·보고서: $d

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin experiment/h64-delta-trim && echo "OK $d" || { echo "FAIL $d"; exit 1; }
done
echo DONE
