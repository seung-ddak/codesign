# CALM·MJX 실험 기록

물리 조건: 로봇 부품 사이 자기 충돌은 끄고, 바닥 충돌은 유지한다.
10초 실행 완료는 정상 명령 추종 승인과 구분한다.

| 실험 | 기록 | 결과 |
|---|---|---|
| Menagerie A1 Isaac Gym 정책 검사 | [보고서](menagerie_calm_check_final/report.html) | LLC 6개·HLC 4개 기록. 형상과 관측 변환 조건 |
| CALM A1 자기 충돌 끔 | [보고서](calm_a1_no_self_collision_six/report.html) | LLC 6개·HLC 4개 기록. 일어나기는 목표 높이에 미도달 |
| 일어나기 시간 연장 | [보고서](calm_a1_stand_5s_diagnostic/report.html) | 5초 연장으로도 충분히 일어나지 않음 |
| 무작위 연속 이동 | [보고서](calm_random_motion_try1/report.html) | 8개 × 10초 완료 |
| 비율 배정 연속 이동 | [보고서](calm_balanced_motion_try1/report.html) | 10개 × 10초 완료 |
| CALM 목표각 MJX 개방 루프 | [보고서](calm_balanced_mjx_try3/report.html) | 10개 모두 전도 조건 발생 |
| Menagerie MJX 폐쇄 루프 | [보고서](menagerie_mjx_policy_try4/report.html) | 접촉 설정 조정 후 10개 × 10초 완료 |
| CALM 원본 MJX 폐쇄 루프 | [보고서](calm_mjx_closed_loop_try1/report.html) | 접촉 설정 조정 후 10개 × 10초 완료 |
| G0·G1 판정 준비 감사 | [보고서](calm_mjx_g01_audit_try1/report.html) | 별도 추종 선별 0/10 통과. 전체 단계 승인 보류 |

## 판정의 한계

추종 선별은 기존 HLC의 속도·방향·기울기 한계를 참고한 별도 검사다.
원래 학습 환경의 평가 절차와 동일하지 않다. 실행 중 바닥 관통 최대 약 18~25mm가 관찰됐다.
관통 허용치, 실제 토크, 정지 시 각속도와 원래 종료 조건은 미검증이다.
MJX 파일의 관절속도·루트 각속도·발 접촉 저장이 부족해 G2 최종 데이터로 승인하지 않았다.

## 파일 구성과 재실행

각 폴더에 영상, HTML 보고서, 수치 요약, 상태 데이터와 가능한 실행 로그를 보관한다.
실패한 시도도 보존하며, 결과가 없는 시도는 실행 로그를 확인한다.
실행 당시 코드 식별 정보가 있는 경우 각 manifest 또는 summary를 우선한다.
외부 체크포인트·원본 CALM 자산과 rl_games 1.1.4 휠은 별도로 준비해야 한다.
외부 라이브러리 휠·캐시·체크포인트는 이 결과 커밋에 포함하지 않는다.

관련 코드: `code/motion_tracking/eval_menagerie_calm.py`,
`check_calm_mjx.py`, `check_menagerie_mjx_policy.py`, `audit_calm_reference.py`.
