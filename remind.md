# TT 확장 연구 — 결정·실험 기록 (remind)

> **정본.** §1 에는 **지금 유효한 결정만** 둔다 (번호는 원래 D 번호 유지). 대체·폐기된 결정과 이전 결과 원문은 `archive/remind_history.md`.
> **서술 규칙 (2026-09-21 사용자 지시): 모든 항목은 결과부터 쓰고 3 줄 이내.** 긴 산문 금지.
>
> 현재 위치 `codesign/remind.md` · 이전 기록 `RL/Transformer_Transformer_연구계획_파이프라인_변천사.md`
> · 원 논문 코드 https://github.com/seung-ddak/transformer-transformer · 기존 BO co-design `quad/daily.md`

---


## 0. 목적 (2026-10-05 개정, D63)

**연구 질문:** 형상·task command·state·action을 함께 학습한 TT로, 새 task reward의 gradient를 형상 토큰에 적용해 **sim 없이** 그 task에 맞는 형상을 생성할 수 있는가.
TT 원본 DSG 구조로 돌아가되, 데이터는 UMI 시연이 아니라 **형상 조건부 보행 정책 π_θ의 rollout**, 조건은 **task command**다. 최종 형상은 반드시 MJX + π_θ로 재채점한다.

| 단계 | 할 일 | 통과 기준 | 상태 |
|------|-------|-----------|------|
| **R1 데이터 수집** | 동결 T5 π_θ로 128 family (Q12) × family 당 continuous 형상 N 개 × 7 command rollout. (형상·command·시작 state·H-step state/action chunk·return) 저장 | 형상 간 return 차이가 seed 분산보다 큼 | **다음 작업** |
| **R2 TT 학습** | command 조건, 형상 + chunk 동시 denoise. 시작 state inpainting 고정 | **held-out 형상 예측 reward ↔ 실제 MJX return 순위상관 ≥ 0.8** | 대기 |
| **R3 기존 reward guidance** | 형상 토큰에만 gradient, 결과 MJX 재채점 | 실제 return > unguided | 대기 |
| **R4 새 reward guidance** | 학습에 안 쓴 reward(command 공간 안)로 sim 0 회 생성 → 재채점 | baseline 비교 우세 (§4) | 대기 |
| **R5 반복 round (co-design)** | guided 형상 주변 π_θ fine-tune → rollout 추가 → TT 재학습 | round 마다 실제 return ↑, 예측-실제 차이 ↓ | 대기 (= Stage 3) |
| **R6 확장** | topology | — | 후속 |

**gate 를 못 넘으면 다음 단계로 가지 않는다.** R2 실패 시 H·N·시작 state 고정부터 바꾸고, return 직접 예측 surrogate 와 비교한다.

---

## 1. 현재 유효한 결정

| # | 결정 |
|---|------|
| D63 | **2026-10-05 연구 방향 (사용자 결정).** 형상 + command + state·action 을 학습한 TT 의 reward gradient 로 형상 생성 (§0 R1~R6). T7 RL 형상 생성(DDPO·one-shot·residual)은 종료 |
| D1·D2 | reference motion 없이 **task 결과가 reward**. noise → denoising → 형상 생성 틀 유지 |
| D7·D13·D14 | 최종 산출물 = **시뮬 상의 (E\*, π\*)**. 시뮬레이터 = **MJX**. 실물 제작·Sim2Real 제외 |
| D35·D49 | 형상 공간 = TT `umi_on_legs_plus_plus` 다리 형상, **팔·레일 제거** (bit 5·팔 위치 uniform 무효). A1 URDF 미사용, legged_gym 은 reward·obs·PPO 구조만 참고 (D37) |
| D36·D44·D45·D47·D50 | 보행 reward = legged_gym 원본 + A1 override + only_positive False + tracking_lin_vel ×3 · tracking_ang_vel ×3 (1.5) |
| D38·D54 | π_θ = 형상 조건부 보행 정책, **T5_try1 checkpoint 동결** (정량 게이트 미달은 한계로 유지) |
| D51·D52 | 평가·채점 = **대칭 고정 명령 7 개**, 각 20 s, DR 끔. 학습 command 는 원본 무작위 range (Q9). 원칙: 원본 따르기 + 가장 단순·빠른 검증 |
| D56 | 채점 단위 = 구체 morphology (choice + continuous). T6 기준: 반복 순위상관 ≥ 0.8, 반복 평균 ≥ T5 seen × 0.9 |
| D55 | 새 작업은 Ponytail full, 착수 시 Archify 다이어그램 + README 를 `artifacts/<작업명>/` 에 둔다 |
| D24 | 동결 π_θ 로 잰 형상 점수는 하한선 (`J(E,θ_고정) ≤ max_θ J`) → R5 에서 완화 |
| 운영 | 학습은 `MUJOCO_GL=egl` 필수 |

---

## 2. 열린 안건

| Q | 안건 | 상태 |
|---|------|------|
| **Q18** | R2 TT: 구조·학습 코드만 재사용해 새로 학습 (A, 권장) vs 사전학습 TT `xrh4wk4l/040.pt` 미세조정 (B, RoboToken·bank 변환 문제 재발) | 미정 — R2 진입 시 결정 |
| **Q19** | RTX 4070 SUPER (12 GB, Ubuntu 22.04, 드라이버 575) 에서 R1~R4 진행 가능 여부 | 조건부 가능 — 시스템 Python 3.8·torch 2.1 대신 uv `.venv` (py3.12) 사용, **RAM ≥ 32 GB** 필요 (컴파일 최대 25~28 GB 실측). RAM 확인 대기, 1024 env 학습 GPU 최대 사용량은 T8_pi_try1 에서 측정 |
| Q6 · Q8 | multi-task reward 합성식 · topology 확장 범위 | R6 이후 |

**결정됨 (10-05):** Q12 R1 family = **128 개 전부** · Q13 R1 = **5 시간 안** (`--hours 4.5`, 형상 수 N 은 처리량으로 결정, seed 1, 에피소드 전체 저장 → H 는 R2 에서 선택) · Q14 task = **① 최고 속도 달리기 ② 최고 높이 점프 ③ 계단 극복** (②③ 은 현재 π_θ 데이터에 없음 → Q15).
**결정됨 (10-05, 2 차):** Q15 = **B** — 통합 π_θ (보행·회전 + 점프 + 계단) 를 먼저 학습한 뒤 R1 을 한 번에 수집. Q16 = **a** — 계단 = legged_gym 원본 rough terrain (5 종 비율·커리큘럼·높이 측정 187 점) 그대로. Q17 = **b** — 점프 = 플래그 command (확률 0.2) + 네 발 공중 보상 + 몸통 상승 보상, 점프 중 lin_vel_z 페널티 끔 (legged_gym 에 없어 D36 예외).
**결정됨 (10-05, 3 차):** π_θ 본 학습 = **A** — T5_try1 규모로 처음부터 (T5 가중치 재사용 안 함). 보행·회전·점프·계단을 정책 하나로 학습 (TaskEnv 가 command·지형을 섞음).

---

## 3. 근거 메모 (D63)

**3.1 왜 다시 gradient 인가.** 과거 RL 을 고른 이유(실제 rollout reward 미분 불가)는 학습된 모델을 거치면 사라진다. T7 DDPO 의 불안정(try6 붕괴)도 피한다.
**3.2 T7 문제와의 관계.** ① 유효 형상 0 → family 고정 + continuous 만 guidance + canonical 표현(choice + uniform 13, bank 미사용) + uniform 범위 clamp 로 원인 제거.
② 같은 advantage → step 별 예측 reward gradient 라 해당 없음 (x̂_0 예측 후 reward, step 별 로그). ③ 분포 밖 최적 → family 안은 R1 uniform 수집으로 해결, family 밖은 R6.
**3.3 새 위험.** ① TT 예측 정확도 (R2 gate) ② π_θ 하한선 J(E,θ_고정) ≤ max_θ J (D24, R5 로 완화) ③ 모델 오차를 파고드는 gradient (전부 MJX 재채점).
**3.4 왜 4족 모션 데이터셋이 아닌가.** TT 원본의 UMI 는 task 명세이고 로봇 움직임은 RL expert 가 만들었다. 우리 대응은 command + T5 π_θ. mocap 은 reference motion 이라 D1 위배.

---

## 4. 실험 규약 (D63)

| 항목 | 값 |
|------|-----|
| 환경 | D27 로컬 GPU · `transformer-transformer/.venv` · MJX 3.10 (이전 V100 규약은 이력) |
| 형상 | `umi_on_legs_plus_plus` 다리 형상, 팔·레일 제거 (D49). **canonical 표현** (choice + uniform 13), RoboToken bank 미사용 |
| 정책 | T5_try1 π_θ 동결 (obs 69, action 12) |
| 명령 | 대칭 고정 7 개, 20 s, DR 끔 (D51·D52) |
| reward | legged_gym 원본 + D44·D45·D47·D50 수정 (T4), task reward 는 Q14 (달리기·점프·계단) |
| run | `stage2/runs/T8/try{N}`, tmux `T8_try{N}` |
| baseline (R3·R4) | family 내 random · unguided TT · **CMA-ES 직접 MJX (같은 sim 예산)** |


---

## 5. 결정 로그 (D63 이후)

| 날짜 | 내용 |
|------|------|
| 2026-10-05 | **D63** TT reward-gradient guidance 로 방향 전환. remind 압축, 폐기 결정·이력 → `archive/remind_history.md` |
| 2026-10-05 | Q12 = 128 family, Q14 = 달리기·점프·계단. 폴더 정리: Stage 1 · 구 Stage 3 · T7 → `archive/{stage1,stage3_coevolution,t7}/`, 미사용 체크포인트 삭제, 동결 π_θ git 추적 |
| 2026-10-05 | Q15 B · Q16 a · Q17 b → `terrain.py`·`task_env.py` (TaskEnv) 구현, T8_pi_smoke (배선 통과, 미보행). π_θ 본 학습 A → T8_pi_try1 시작. Q18 (R2 TT 초기화)·Q19 (4070 SUPER) 신설 |

---

## TODO

### 완료 요약 (상세 원문 `archive/remind_history.md`)

| # | 결과 |
|---|------|
| T1 Stage 1 | TT 행 재현 (Pos Err 3.0 cm · Survival 99.4 %), RL guidance 치환 동작 확인 |
| T3 env | `stage2/code/locomotion_env.py` obs 69 (48 + 형상 21) · action 12. 발만 충돌해 넘어지면 몸통이 바닥 통과 |
| T4 reward | 종료 = 몸통 접촉만, thigh·calf 는 collision 페널티. 수정 이력은 §1 D36 행 |
| T5 π_θ | T5_try1 128 choice 생존 84 % · 이동 69 % · 회전 56 %, 영상 확인 후 동결 (`stage2/runs/T5/try1`) |
| T6 채점기 | 반복 순위상관 0.913, 이동 0.686 · 회전 0.585 · 생존 0.841 → 통과. choice 순위상관 0.374 (진단값) |
| T7 RL 형상 생성 | DDPO try6 update 26 부터 유효 형상 0/32 → 갱신 정지, return 추세 없음. 종료 (D63), 코드 `archive/t7/` |

### T8. TT reward-gradient guidance (D63) — 통합 π_θ 준비 중

**env 구현 (10-05):** `terrain.py` (legged_gym 지형 재구현) · `task_env.py` (TaskEnv: heightfield·높이 측정·커리큘럼·점프, obs 257) · `train_locomotion_policy.py --env task`. `test_task_env.py` 통과.
MJX 가 cylinder–heightfield 충돌 미지원 → thigh·calf 접촉 감지 cylinder 를 같은 크기 capsule 로 교체. 점프 보상 scale (air 1.0 · height 5.0) 은 임시값.
**T8_pi_smoke (10-05): 배선 통과, 아직 못 걸음.** 512 env · 122.9 만 step (T5 의 1/20, T5 도 300 만 step 에선 못 걸음), 평지 전진 생존 2 %, 점프는 앉아 버팀. 컴파일 학습 41 분·평가 37 분. [보고서](https://claude.ai/artifact/TveNFVzzGnbycrcEf3SDrB)
**T8_pi_try1 (10-05 17:01 시작, 사용자 결정 A):** T5_try1 과 같은 규모 처음부터 — 1024 env · choice 당 seed 8 (1024 형상) · 2,457.6 만 step · checkpoint 10, `--env task` (보행·회전·점프·계단 정책 하나). tmux `T8_pi_try1`.
다음: 결과 확인 → `collect_rollouts.py` 를 TaskEnv 용으로 수정 → R1. `test_locomotion_env.py` 의 collision 단언은 D44 이후 갱신 안 된 기존 실패.
단계·gate 는 §0 표를 따른다.
