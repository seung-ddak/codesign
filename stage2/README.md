# Stage 2 — 동결 π_θ + 형상 생성

코드 `code/`, 결과 `runs/<run>/`. 명령은 전부 **`codesign/` 에서** 실행한다 (경로가 `stage2/...` 기준).
결정·결과 정본은 `../remind.md`. T7(RL 형상 생성)은 D63 으로 종료, 코드는 `../archive/t7/code/`. 현재 T8 (TT reward-gradient guidance).

## 흐름

T5 π_θ 학습 → T6 평가기 검증 → (T7 RL 시도, 종료) → T8 R1 rollout 수집 → R2 TT 학습 → guidance.

```sh
P=../transformer-transformer/.venv/bin/python
S=../scripts/tmux_run.sh                      # rl_scripts/ 를 codesign 옆 scripts/ 로 복사해 둔 것 (SETUP.md)

# T5 π_θ 학습 + 후처리 체인 (평가·영상·보고서)
TMUX_RUN_CWD=$PWD bash $S T5_tryN "$P stage2/code/train_locomotion_policy.py --out stage2/runs/T5/tryN --timesteps 24576000 --checkpoints 10"   # T5_try1 조건 (= config.json)
TMUX_RUN_CWD=$PWD bash $S T5_tryN_post "bash stage2/code/chain_after_train.sh T5_tryN stage2/runs/T5/tryN stage2/runs/T5/tryN 'T5 π_θ 선학습' 'T5 tryN'"

# T5 게이트 (eval json 만 읽음, 수 초)
$P stage2/code/t6_evaluator_check.py t5 --run stage2/runs/T5/try1

# T6 평가기 검증 (~50 분) → 보고서
TMUX_RUN_CWD=$PWD bash $S T6_tryN "$P stage2/code/t6_evaluator_check.py t6 --run stage2/runs/T5/try1 --out stage2/runs/T6/tryN"
$P stage2/code/make_stage_report.py t6 --run stage2/runs/T6/tryN --title T6_tryN

# T8 통합 π_θ (TaskEnv: 지형·높이 측정·점프) — smoke 후 본 학습
TMUX_RUN_CWD=$PWD bash $S T8_pi_smoke "$P stage2/code/train_locomotion_policy.py --env task --out stage2/runs/T8/pi_smoke --seeds-per-choice 1 --num-envs 512 --timesteps 1228800 --checkpoints 2"

# T8 R1 rollout 수집 (TaskEnv 용 수정 전, 현재는 flat π_θ 용) — 시간 예산 안에서 batch 반복 (batch 1 개 = 128 choice × 새 형상 1 개 × 명령 7 개)
TMUX_RUN_CWD=$PWD bash $S T8_smoke "$P stage2/code/collect_rollouts.py --run stage2/runs/T5/try1 --out stage2/runs/T8/smoke --steps 50 --max-batches 2"
TMUX_RUN_CWD=$PWD bash $S T8_try1 "$P stage2/code/collect_rollouts.py --run stage2/runs/T5/try1 --out stage2/runs/T8/try1 --hours 4.5"

```

## 코드

| 파일 | 하는 일 |
|---|---|
| `locomotion_env.py` | T3 env (TT 형상·액추에이터, obs 69, action 12) + T4 reward·종료 |
| `terrain.py` | legged_gym rough terrain 재구현 (Q16 a). `python terrain.py` = 자체 검사 |
| `task_env.py` + `test_task_env.py` | T8 TaskEnv — heightfield·높이 측정 187 점·지형 커리큘럼·점프 (Q17 b), obs 257 |
| `multi_design_env.py` | 구조 그룹별 MJX vmap, `structure_key` (레일 제거로 bit 5 무효) |
| `train_locomotion_policy.py` | π_θ PPO (brax). `--env flat` = T5, `--env task` = T8 통합. run 폴더에 config·designs·progress·checkpoints |
| `eval_locomotion_policy.py` | 고정 명령 7 개 평가 (D51·D52, DR 끔) → `eval_<step>.json`, `view` 는 MuJoCo 뷰어 |
| `record_eval_video.py` | 평가 순위별 영상 (무작위 / `--fixed` 고정 명령) |
| `make_report.py` + `report_template.html` + `report_style.css` | T5 보고서 (`timeline.json` 이 서술부) |
| `t6_evaluator_check.py` | T5 게이트 (`t5`) / T6 검증 (`t6`, D56 기준) |
| `design_scorer.py` | 채점기 — 동결 π_θ 로 임의 형상 (choice + uniforms) return, 1 회 컴파일. T8 R1 수집의 기반 |
| `collect_rollouts.py` | T8 R1 — 동결 π_θ rollout 을 step 단위로 저장 (`batch_XXXX.npz`, git 미추적), `--hours` 예산 안에서 반복 |
| `make_stage_report.py` | T6 보고서 (T7 모드는 이력용) |
| `chain_after_train.sh` | T5 학습 후 평가·영상·보고서 체인. 대기는 `tmux has-session -t =이름` (정확 일치) |
| `test_locomotion_env.py` · `test_multi_design_env.py` | env 동작 확인 (새 PC 첫 점검용) |

## run · 보고서 (아티팩트)

각 run 의 `report.html` 이 아티팩트 원본이다. 링크는 claude.ai 비공개 페이지 (소유자만 열림).
체크포인트는 T5_try1 의 마지막 것(`000024576000`)만 git 에 있다 (동결 π_θ). T7 ψ 체크포인트는 삭제했다 (2026-10-05).

| run | 결과 | 보고서 | 아티팩트 |
|---|---|---|---|
| `t5_a1_nopos_1000it` | 버티기 정책 (D45 근거) | `report.html` · `flow.html` | [발자취](https://claude.ai/artifact/3T7nYqhuFMjwFpbafHG4p3) · [흐름](https://claude.ai/artifact/GNC4pYyfDqK9nN6guGPg6C) |
| `t5_noarm_200it` | 팔 제거 시범, 생존 하락 (D47 근거) | `report.html` · `flow.html` | [보고서](https://claude.ai/artifact/B38EyhgpLwX5ZAnJoZaH3g) · [흐름](https://claude.ai/artifact/E6vDEqufAgckEP3JqTwp25) |
| `t5_norail_x3_1000it` | 첫 보행 성공 (16 choice), 파일럿 π_θ | `report.html` · `flow.html` | [보고서](https://claude.ai/artifact/RCiJo8zL1BAqh29UKoyBgQ) · [흐름](https://claude.ai/artifact/34tHVv4EdUsH7karioNpDj) |
| `T5_try1` | 128 choice, 생존 84 %·이동 69 %·회전 56 % → 게이트 미달 | `report.html` | [보고서](https://claude.ai/artifact/QNH9jqS4ocGjo3nYe6z1k4) |
| `T6_smoke` | 파일럿 π_θ 로 T6 스크립트 점검, 불합격 (커버리지) | `report.html` | [보고서](https://claude.ai/artifact/4Ug1HfYLNPNjJTTbvikkhk) |
| `T6_try1` | D56 기준 통과 (반복 순위상관 0.913) | `report.html` | — |
| `T7_smoke` | 구 residual 방식의 과거 연결 검사; DDPO 근거 아님 | `report.html` | [보고서](https://claude.ai/artifact/JGnqcAA2NJkyyJn2WVTpvv) |
| `T7_try1` | 구 residual 방식·파일럿 π_θ 구조 키 부족 → ValueError | `report.html` | [보고서](https://claude.ai/artifact/4gFyWk2r7EqmMu3NXL3TqT) |
| `T7_smoke2` | DDPO 3 업데이트 완주, return 62.4→54.6→59.8 추세 없음, 업데이트당 ~55 분 | `report.html` · `videos/` | [보고서](https://claude.ai/artifact/5Mpj1Rzfjr1jT4UDQR3mu7) |
| `T7_try6` | DDPO 50 업데이트 실패. update 26 부터 유효 형상 0/32 (bank 변환 불가), ψ 갱신 정지. 붕괴 전 return 59.5→60.7 추세 없음 | `report.html` · `videos/` | [보고서](https://claude.ai/artifact/27KrVahfUiphGdysvepKEL) |
| `T8_pi_smoke` | TaskEnv (지형·점프) 배선·처리량 smoke, 122.9 만 step — 아직 걷지 못함 (T5 의 1/20), 컴파일 41 분 | `report.html` · `videos/` | [보고서](https://claude.ai/artifact/TveNFVzzGnbycrcEf3SDrB) · [흐름](https://claude.ai/artifact/C6sn9btvjUmm1dnUZ3Mp9z) |
| `runs/A1_tracking` | A1 몸 레퍼런스 tracking (try1·try2·topo1100) 통합 | `report.html` | [보고서](https://claude.ai/artifact/3rmR88rdXEev4q9Tx9Z58W) |
| `runs/tiltYaw_test` | tiltYaw 넘어짐 시험 (penalty·형상·kp·발 고정) 통합. 결론 = tiltYaw 제외 | `report.html` | [보고서](https://claude.ai/artifact/Uz7BNdsam4gd59D35VXDJK) |
| `runs/TT_retarget` | A1 → TT 4 형상 IK 리타겟 + 전환 3 종. 데이터 본체 (artifacts/retarget_tt 는 링크) | `report.html` | [보고서](https://claude.ai/artifact/M2VSnGMLch57wSFDLHJxdw) |
| `runs/TT_tracking` | TT 4 형상 tracking (tt00000010 try1~3, tt_js100M, tt_trans50M, tt_trans100M 진행 중) 통합 | `report.html` | [보고서](https://claude.ai/artifact/EXm61K5fNm9ATibCN5c9fn) |

그 외 `t5_screen_200it` · `t5_smoke_*` · `t5_a1_1000it` · `t5_lin5_200it` 는 T4/T5 조정 과정 기록 (remind D43~D47).
