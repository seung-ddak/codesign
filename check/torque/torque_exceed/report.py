import sys; sys.path.insert(0, '/home/cgr-ugrad-2026/quad_ws/codesign/chain/code')
from pathlib import Path
from common import report, img
H = Path(__file__).parent
report(H / 'report.html', '토크 초과: CALM 모션 × 학습 형상 9 개', [
  ('결론', '<ul><li><b>모션 기준 6,000 개</b> (모션마다 형상 2 개에 리타겟, 둘 중 더 나쁜 쪽): 한 프레임이라도 초과 <b>3,965 (66 %)</b>, 초과 프레임 >1 % 2,401 (40 %), >5 % 789 (13 %). 두 형상 모두 초과 2,395 (40 %). 최대 토크 120 N·m 이상 약 850 개.</li><li>짝 기준 12,000 쌍 중 한 프레임이라도 초과 <b>6,360 (53 %)</b>, 초과 프레임 >1 % 3,414 (28 %), >5 % 991 (8 %).</li>'
   '<li>초과는 거의 <b>thigh 관절</b> (0.27–0.47 %), hip·calf 는 0.02–0.11 %.</li>'
   '<li>모션 속도 0.3 m/s 까지 >1 % 초과는 9–17 %, 0.4 m/s 부터 급증 (0.5–0.6 m/s 64 %, 0.6–0.8 m/s 81 %).</li>'
   '<li>다리 긴 형상 (l+20) 66–81 %, 짧은 형상 (l-20) 29–34 %. 묶음 중 trotForwardSlow 90 %.</li>'
   '<li>S7: 토크 초과 없는 CALM 통과 86 %, 초과 1–5 % 는 29 %.</li></ul>'),
  ('모션 기준 (6,000 개, 형상 2 개 중 더 나쁜 쪽)', img(H / 'motion_counts.png', '묶음별 초과 모션 수, 배정된 형상 2 개 중 초과한 형상 수')
   + img(H / 'motion_tau_speed.png', '모션별 최대 토크 분포 (120 이상은 마지막 칸) · 속도별 초과 모션 비율')),
  ('형상별 · 묶음별 (모션 × 형상 짝 12,000 기준)', img(H / 'exceed_counts.png', '모션 수, 초과 프레임 비율 단계별')),
  ('최대 토크 분포', img(H / 'tau_max_hist.png', '모션별 최대 |토크|, 점선 = 한계 33.5 N·m')),
  ('관절별 · 속도별', img(H / 'joints_speed.png', '왼쪽: 관절별 초과 프레임 비율. 오른쪽: A1 레퍼런스 평균 이동 속도 구간별 초과 모션 비율')),
  ('토크 초과와 S7 추종', img(H / 's7_relation.png', '오른쪽 그래프 = 토크 초과 프레임 비율별 S7 통과율'))],
  [('데이터', 'chain/runs/s5_retarget/references/<형상>/00000011/*.npz (리타겟, 30 Hz), assignment.json 의 형상·모션 짝 12,000'),
   ('토크', 'pd_replay.inverse_check 역동역학 다리 토크 12 축 (inv_tau), 한계 33.5 N·m (리타겟·학습과 같음)'),
   ('S7', 'chain/runs/s7_eval/results.json, 통과 = 생존 · 관절 RMSE ≤ 10° · 몸통 RMSE ≤ 10 cm')])
