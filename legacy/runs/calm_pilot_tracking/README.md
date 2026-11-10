# CALM 레퍼런스 TT 물리 추종

| 시도 | 조건 | 결과 | 보고서 |
|---|---|---|---|
| try1 | 30Hz·발 고정 없음·기존 위치 PD | 생존 5/5, 오차 선별 0/5 | [보고서](try1/report_try1.html) |
| try2 | 동일 레퍼런스·PPO 5,246,976 스텝 | 생존 5/5, 오차 선별 0/5 | [비교 보고서](try2/report_try2.html) |
| try3 | 같은 PPO·MJX·발/바닥 미끄럼 마찰 1.0 | 생존 5/5, 오차 선별 0/5 | [마찰 비교 보고서](try3/report_try3.html) |

| try4 | 같은 e000 PPO·새 형상 5개·MJX·재학습 없음 | 생존 19/25, 오차 선별 0/25 | [형상 전이 보고서](try4/report_try4.html) |

try2는 MJX GPU에서 정책 하나를 학습하고, try1과 같은 MuJoCo CPU에서 평가했다. 두 시도의 측정값과 영상 비교는 try2 보고서를 참조한다.
