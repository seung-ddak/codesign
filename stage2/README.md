# Stage 2 — 동결 π_θ + DDPO 형상 생성

코드 `code/`, 결과 `runs/<run>/`. 명령은 전부 **`codesign/` 에서** 실행한다 (경로가 `stage2/...` 기준).
결정·결과 정본은 `../remind.md` (D43~D53, TODO T5~T7).

## 흐름

T5 π_θ 학습 → T6 평가기 검증 → T7 TT denoiser ψ DDPO 학습.

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

# T7 DDPO + 보고서 (새 방식의 전체 update 시간은 첫 실행에서 측정)
TMUX_RUN_CWD=$PWD bash $S T7_tryN "bash stage2/code/run_t7.sh stage2/runs/T5/try1 stage2/runs/T7/tryN 100"
```

T7 메모리는 `--global-batch 32 --generation-microbatch 4 --ddpo-minibatch 32`로 조절한다. `--slots`는 MJX 평가 병렬도이며 별도다.

T7 의 `--pi-theta` 는 **16 구조 키를 모두 학습한 run** 이어야 한다 (파일럿 `t5_norail_x3_1000it` 은 2 종뿐 → T7_try1 ValueError).

## 코드

| 파일 | 하는 일 |
|---|---|
| `locomotion_env.py` | T3 env (TT 형상·액추에이터, obs 69, action 12) + T4 reward·종료 |
| `multi_design_env.py` | 구조 그룹별 MJX vmap, `structure_key` (레일 제거로 bit 5 무효) |
| `train_locomotion_policy.py` | T5 π_θ PPO (brax). run 폴더에 config·designs·progress·checkpoints |
| `eval_locomotion_policy.py` | 고정 명령 7 개 평가 (D51·D52, DR 끔) → `eval_<step>.json`, `view` 는 MuJoCo 뷰어 |
| `record_eval_video.py` | 평가 순위별 영상 (무작위 / `--fixed` 고정 명령) |
| `make_report.py` + `report_template.html` + `report_style.css` | T5 보고서 (`timeline.json` 이 서술부) |
| `t6_evaluator_check.py` | T5 게이트 (`t5`) / T6 검증 (`t6`, D53 기준 ①②③) |
| `design_scorer.py` | T7 채점기 — 동결 π_θ 로 임의 TT 형상 return, 1 회 컴파일 |
| `ddpo_hardware.py` | stochastic DDIM transition 기록과 clipped DDPO-IS |
| `train_hardware_guidance.py` | T7 DDPO 실행 (trainable TT ψ + frozen πθ evaluator) |
| `train_oneshot_morphology.py` | T7 one-shot RL — valid topology + 13 uniforms 직접 생성, frozen πθ 채점 |
| `train_rl_denoiser_guidance.py` | T7 RL guidance — 동결 TT denoiser의 DSG gradient 자리를 `πφ` residual action으로 교체 |
| `make_stage_report.py` | T6·T7 보고서 |
| `chain_after_train.sh` · `chain_t6_t7.sh` · `run_t7.sh` | 자동 체인. 대기는 `tmux has-session -t =이름` (정확 일치) |
| `diag_fall_modes.py` | 무입력 vs 무작위 행동 낙상 진단 |
| `test_locomotion_env.py` · `test_multi_design_env.py` | env 동작 확인 (새 PC 첫 점검용) |

## run · 보고서 (아티팩트)

각 run 의 `report.html` 이 아티팩트 원본이다. 링크는 claude.ai 비공개 페이지 (소유자만 열림).
체크포인트는 T5_try1 · t5_norail_x3_1000it 의 **마지막 것만** git 에 있다 (T6·T7 이 동결 π_θ 로 씀).

| run | 결과 | 보고서 | 아티팩트 |
|---|---|---|---|
| `t5_a1_nopos_1000it` | 버티기 정책 (D45 근거) | `report.html` · `flow.html` | [발자취](https://claude.ai/artifact/3T7nYqhuFMjwFpbafHG4p3) · [흐름](https://claude.ai/artifact/GNC4pYyfDqK9nN6guGPg6C) |
| `t5_noarm_200it` | 팔 제거 시범, 생존 하락 (D47 근거) | `report.html` · `flow.html` | [보고서](https://claude.ai/artifact/B38EyhgpLwX5ZAnJoZaH3g) · [흐름](https://claude.ai/artifact/E6vDEqufAgckEP3JqTwp25) |
| `t5_norail_x3_1000it` | 첫 보행 성공 (16 choice), 파일럿 π_θ | `report.html` · `flow.html` | [보고서](https://claude.ai/artifact/RCiJo8zL1BAqh29UKoyBgQ) · [흐름](https://claude.ai/artifact/34tHVv4EdUsH7karioNpDj) |
| `T5_try1` | 128 choice, 생존 84 %·이동 69 %·회전 56 % → 게이트 미달 | `report.html` | [보고서](https://claude.ai/artifact/QNH9jqS4ocGjo3nYe6z1k4) |
| `T6_smoke` | 파일럿 π_θ 로 T6 스크립트 점검, 불합격 (커버리지) | `report.html` | [보고서](https://claude.ai/artifact/4Ug1HfYLNPNjJTTbvikkhk) |
| `T6_try1` | 시작 직후 중단, 재실행 대기 (`designs.json` 만) | — | — |
| `T7_smoke` | 구 residual 방식의 과거 연결 검사; DDPO 근거 아님 | `report.html` | [보고서](https://claude.ai/artifact/JGnqcAA2NJkyyJn2WVTpvv) |
| `T7_try1` | 구 residual 방식·파일럿 π_θ 구조 키 부족 → ValueError | `report.html` | [보고서](https://claude.ai/artifact/4gFyWk2r7EqmMu3NXL3TqT) |
| `T7_smoke2` | DDPO 3 업데이트 완주, return 62.4→54.6→59.8 추세 없음, 업데이트당 ~55 분 | `report.html` · `videos/` | [보고서](https://claude.ai/artifact/5Mpj1Rzfjr1jT4UDQR3mu7) |
| `T7_try6` | DDPO 50 업데이트 실패. update 26 부터 유효 형상 0/32 (bank 변환 불가), ψ 갱신 정지. 붕괴 전 return 59.5→60.7 추세 없음 | `report.html` · `videos/` | [보고서](https://claude.ai/artifact/27KrVahfUiphGdysvepKEL) |

그 외 `t5_screen_200it` · `t5_smoke_*` · `t5_a1_1000it` · `t5_lin5_200it` 는 T4/T5 조정 과정 기록 (remind D43~D47).
