# 실험 결과 찾기

현재 작업과 이전 실험을 이 페이지에서 구분한다.

| 용도 | 위치 |
|---|---|
| CALM 연속 모션 10,000개 수집 | [설명](calm_motion_10,000/README.md) · [보고서](calm_motion_10,000/dataset/report.html) · [실시간 상태](calm_motion_10,000/dataset/status.json) |
| TT 디노이징 내부 PPO, E/S/A 공동 생성 | [보고서](pipeline_v1/rl_denoising_v1/report.html) |
| 행동 PPO 보상 설계 실험 | [v2 보고서](pipeline_v1/task_rl_v2/report.html) |
| TT v16 로컬 평가·guidance 분석 | [보고서](pipeline_v1/v16_local_execution/report.html) |
| 세 형상 모션 트래킹 비교 | [보고서](calm_pilot_tracking/try2/report_try2.html) |
| 전체 실험 목록 | [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) |
| 종료·중단 실험 압축본과 복원 방법 | [정리 기록](../archive/experiment_cleanup_20261103/README.md) |

2026-11-03 정리: 사용 중인 학습 데이터와 정책은 원래 경로에 유지한다. 종료된 일부 시험은 압축 보관했다. 과거 문서의 원래 경로가 필요하면 정리 기록에서 해당 묶음을 복원한다. 컴파일된 .mjb도 압축본에서 원래 바이너리로 복원할 수 있다.

[추가 코드·smoke 정리 내역과 복원](../archive/code_cleanup_20261103/README.md): 종료된 v7 실행 코드·전용 테스트·임시 시험 출력을 보관했다.

[1,000형상 × 100모션 dynamics 수집](calm_dynamics_1000x100/README.md): 공통30개 + 형상별 중복 없는70개, 추가모션 재사용7~8회로 균등 분산.
