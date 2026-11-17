# 연속 실험 체인 (2026-11-12)

목표: 모션 트래킹 정책 → 연속 형상 rollout → 생성 모델 (TT) 학습. 단계마다 GitHub 이슈 (seung-ddak/codesign) 에 HTML 보고서 댓글.

| 단계 | 파일 | 내용 | 결정 (이슈 댓글 `승인 key=value`) |
|---|---|---|---|
| 0 | chain.py | 짧은 다리 실험 (check/short_leg) 종료 대기 | |
| 1 | s1_shape_graph.py | 지표 대 형상 그래프. 다리 하한 = L20 (사용자 결정 11-12), 몸통 폭 변경 없음 | `legs=-20,0,20 bodies=-20,0,20` |
| 3 | s3_generalization.py | 기존 E0~E4 정책을 학습하지 않은 형상에 적용 (체인 밖, 먼저 실행) | |
| 4 | s4_calm.py, calm_joystick.py, collect_calm_joystick.py | 조이스틱 방식 CALM 모션, trot Slow 레퍼런스 6 개 (전·후·좌·우·CW·CCW) 조합, 속도 배율 0.5~2.5, 묶음당 1,000 | |
| 5 | s5_retarget.py | 학습 형상 격자 리타겟 (30 Hz, 레퍼런스는 50→30 Hz 재표본): 레퍼런스 25 = 모든 형상, CALM = 모션당 형상 2 개 | |
| 6a | s6a_ref.py | 레퍼런스 1 개 × 학습 형상 전부, 정책 6 개 (약 5M 씩, 6 단계와 같은 설정) | |
| 6 | s6_train.py | 묶음별 공유 PPO (형상 토큰 입력, 동시 환경 1,024, lr 1e-3, 30 Hz 제어 / 240 Hz 물리) | |
| 7 | s7_eval.py | 마지막 5 체크포인트 중 선택 + 학습/비학습 짝 평가 (생존 · 관절 ≤ 10° · 몸통 ≤ 10 cm) | `n_shapes=20 per_group=100` |
| 8 | s8_collect.py | 학습하지 않은 연속 형상 rollout (rollout_data 형식) | `tt_args=--horizon_64_...` (공백 대신 `_`) |
| 9 | s9_tt.py | train_tt.py | |

- 실행: `tmux new-session -d -s chain "bash -c 'cd codesign && source code/env_osmesa.sh && ../transformer-transformer/.venv/bin/python -u chain/code/chain.py >> chain/logs/chain_run.log 2>&1'"`
- 폴더: 코드 `code/`, 결과 `runs/`, 로그 `logs/` (chain.log, chain_run.log). 상태: `state.json` (끝난 단계, 결정 값, 이슈 번호), 다시 실행하면 이어서 한다.
- 보고서: `reports` 브랜치 `<날짜>/<단계>/report.html` (contents API, 로컬 git 미사용). 댓글에 요약 그림 + 보고서 링크.
- 승인: 소유자·협업자 댓글 `승인` / `승인 key=value ...` / `중단`, 10 분마다 확인.

- 2026-11-17 runs 정리: `runs/s6_tracking/` (s6_train, s6a_ref, s6c_calm, s6c50_calm), `runs/s8_rollout/` (s8_collect; s8_all 은 2026-11-17 중단·삭제), `runs/s9_diffusion/` (s9_tt, s9_tt_200k, s9_tt_m4). 옛 경로는 심볼릭 링크로 남겨 코드·저장된 절대 경로가 그대로 동작.
