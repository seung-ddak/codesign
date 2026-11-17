import sys; sys.path.insert(0, '/home/cgr-ugrad-2026/quad_ws/codesign/chain/code')
from pathlib import Path
from common import report, img
H = Path(__file__).parent
report(H / 'report.html', '방향성 지표: 수행 가능 모션 통과율 · 통과 rollout 의 속도 공간 범위', [
  ('결론', '<ul><li>수행 가능 CALM (불가능 프레임 ≤1 %) 통과 <b>97/107 (91 %)</b>, 불가능 프레임 >1 % 는 72/163 (44 %). 전체 169/270 (63 %).</li>'
   '<li>통과 rollout 이 덮는 속도 칸: A1 레퍼런스만 <b>14</b> → A1+CALM <b>355</b> (25 배). 평가한 CALM 전체 481 칸 중 74 %.</li>'
   '<li>정책은 수행 가능한 모션은 잘 따라감. 실패는 대부분 수행 불가능 구간이 있는 모션.</li></ul>'),
  ('지표 2: 수행 가능성별 통과율', img(H / 'm2_feasible_pass.png', '파랑 = CALM, 주황 = A1 레퍼런스. 막대 위 = 통과/전체')),
  ('지표 3: 속도 공간 범위', img(H / 'm3_coverage.png', '1 초 창마다 몸통 기준 평균 (vx, vy, wz). 칸 = 0.1 m/s × 0.1 m/s × 0.2 rad/s'))],
  [('평가 결과', 'chain/runs/s7_eval/results.json (묶음별 공유 정책 100M, 형상 9, 324 개)'),
   ('통과 기준', '끝까지 생존 · 관절 RMSE ≤ 10° · 몸통 RMSE ≤ 10 cm'),
   ('수행 가능성', '리타겟 역동역학 (pd_replay.inverse_check): 다리 |토크| > 33.5 N·m 또는 몸통 잔차 힘 > 체중 10 % 인 프레임 비율'),
   ('속도', 'A1 레퍼런스 (s5_retarget/a1_reference, 형상 무관) 의 몸통 위치·yaw 미분')])
