# TT 확장 연구 — 결정·실험 기록 (remind)

> **정본.** 결정은 §1 에 D 번호로, 실험은 §4 규약대로, 결과는 §5 에 넣는다.
> **서술 규칙 (2026-09-21 사용자 지시): 모든 항목은 결과부터 쓰고 3 줄 이내.** 긴 산문 금지.
>
> 현재 위치 `codesign/remind.md` · 이전 기록 `RL/Transformer_Transformer_연구계획_파이프라인_변천사.md`
> · 원 논문 코드 https://github.com/seung-ddak/transformer-transformer · 기존 BO co-design `quad/daily.md`

---

## 0. 목적

TT 의 "noise → denoising → 형상" 틀은 유지하되, reference motion + reward gradient guidance 를
**실제 locomotion task 수행 결과를 reward 로 삼는 RL guidance** 로 확장한다. 최종 산출물은 시뮬 상의 (E*, pi*).

### 현재 연구 구조 — 단계별 검증 (2026-09-25 개정)

| 단계 | 무엇을 하나 | 핵심 질문 | 상태 |
|------|-------------|-----------|------|
| **1 단계 · RL guidance 성립** | 고정 TT prior + UMI tracking 조건에서 DSG 자리에 RL guidance 적용 | RL guidance 로 형상이 생성되고 tracking 이 유지되는가 | **완료** |
| **2 단계 · task-result hardware generation** | TT design space 에서 locomotion policy 를 먼저 학습·고정하고 pi_phi 만 task return 으로 학습 | reference motion 없이 실제 locomotion 결과로 좋은 형상을 생성하는가 | **현재 우선순위** |
| **3 단계 · hardware-policy co-design** | 2 단계 통과 후 pi_phi 와 pi_theta 를 분리된 timescale 로 함께 최적화 | 형상과 정책을 함께 개선할 수 있는가 | 후속 |
| **확장** | multi-task → broad topology design space | 하나의 hardware + 하나의 multi-task policy 로 확장 가능한가 | 후속 |

**현재 우선순위: 2 단계 단순 실험부터 진행한다 (2026-09-25 개정).** 과거 D12·D26의 공진화 경로는 폐기하지 않고 3 단계로 보류한다.
형상은 TT `umi_on_legs_plus_plus` design space 와 MJX 경로를 유지하고, A1 URDF 로 교체하지 않는다.

**현재 2 단계의 한 round:** ① 고정 TT prior + pi_phi residual 로 morphology 생성 → ② 동결된 morphology-conditioned pi_theta 로 평지 locomotion rollout → ③ legged_robot 원본 reward 의 episode return 계산 → ④ 같은 형상을 여러 seed 로 평가 → ⑤ return 으로 pi_phi 갱신 → ⑥ 다음 round.
pi_theta 는 2 단계 시작 전에 TT design space 의 다양한 유효 morphology 로 충분히 학습한 뒤 고정한다.

---

## 1. 확정된 결정

| # | 결정 |
|---|------|
| D1 | reference motion 사용 안 함 → **task 성공 결과 자체가 reward** |
| D2 | noise → denoising → hardware 생성 **틀 유지** |
| D3 | task 별 로봇 X → **하드웨어 1 대 + multi-task policy 1 개** |
| D4 | design space 에 topology 포함(다리 수·DoF) — 2 단계로 보류 |
| D5 | ∇_E R 대신 **RL policy 가 denoising 방향 결정** |
| D6 | **surrogate 불필요.** 배치 rollout + denoising 2 회로 설계당 12.3 s (§5.4) |
| D7 | 최종 산출물 = **시뮬레이션 상의 (E\*, π\*)** |
| D8 | 1 단계는 **TT design space·prior 그대로**. 이산 128 조합은 expert 128 개로 전부 평가 가능 (§5.10) |
| D9 | task 는 모션 추종이 아니라 **결과 기반** (넘어지지 않고 목표 도달 = 성공) |
| D10 | ① 무조건부 생성 ② 성공 = **제한 시간 내 목표 도달 AND 무낙상** ③ **팔 고정**, 다리·몸통만 탐색 |
| D11 | RL action = **residual guidance**. denoiser 고정, `x_{k-1} = denoise(x_k) + π_φ(s_k)` |
| D12 | **공진화.** `π_θ(o,E)` 가 곧 최종 정책, π_φ 와 같은 reward 로 번갈아 성장. **2 단의 구성요소** (2026-09-21 승격) |
| D13 | 시뮬레이터 = **MJX/MJWarp**. legged_gym 은 walk_N 전용 동결 |
| D14 | **실물 제작·Sim2Real 제외.** 논문용 baseline·novelty 정리는 후순위 |
| D15 | task set = ① 보행 ② 계단 상승 ③ 계단 하강 ④ 불균형 지형 ⑤ 점프 |
| D16 | **과거 고정 ctrl 실험용 결정. 현재 공진화에는 적용하지 않는다** (D12·D26). |
| D17 | **L0~L3 사다리 폐기 (2026-09-21).** 비교는 **TT** (논문 그대로) vs **RL** (우리 guidance) 둘만. §0 목적에 직결 |
| D18 | **평가 프로토콜 = 논문 §3 그대로.** 평가기 = choice 별 **RL expert**, seed **9** + 95 % CI, held-out UMI 궤적 |
| D19 | **1 단 측정 지표 = Pos Err · Orn Err · Survival · Reward · 최적화 시간.** 전진 거리는 쓰지 않는다 (§5.16) |
| D20 | **종료 조건 = 낙상 또는 ε_p > 0.8 m** (논문 quadruped 기준). 자체 낙상 판정식은 쓰지 않는다 |
| D21 | **"TT 그대로" = `evaluate_hardware_opt_rl.py` + `rl_policies.zip`** (TT prior + DSG + choice 별 expert) |
| D22 | **2 단 reward 확정 (2026-09-21)**: 낙상 = TT 의 `penalty_bad_contact_pairs` 링크 접촉 **또는** up_vec·z < 0.1 → 즉시 종료 / 점수 = 종료 전 전진 진척도 (목표 **1.0 m** 정규화) / 성공 = 종료 없이 도달. π_φ·π_θ 가 같은 이 reward 로 학습 |
| D23 | **"형상에 맞는 최적 정책" 은 저절로 생기지 않는다.** π_θ 를 D22 reward 로 같이 학습시킬 때만 생긴다 → D12 가 필수 |
| D24 | **π_θ 고정으로 잰 형상 점수는 하한선이다** (`J(E,θ_고정) ≤ max_θ J`). **1 단에만 해당** — 2 단은 D12 로 이 한계를 푼다 |
| D25 | **π_θ 는 형상 조건부 + 2-timescale.** 가중치·관측 정규화는 형상과 round 사이에 이어 간다. 새 보행 관측이 TT expert 와 달라 이번 경로는 fresh 초기화 (D26). |
| D26 | **2026-09-22 공진화 우선.** reward = 낙상 전 월드 +x 변위 증가 / 1.0 m, 후진은 음수; 지정 링크 접촉 또는 up_z < 0.1 에서 종료. π_θ 는 step reward, π_φ 는 같은 reward 의 rollout 합 사용. |
| D27 | **2026-09-22 공진화 실행 환경.** `/home/cgr-ugrad-2026/quad_ws/transformer-transformer/.venv` · Python 3.12.14 · RTX 4070 SUPER 12 GB · CUDA 12.6 · JAX 0.10.2 · MuJoCo 3.10.0 · Brax 0.14.2. 기존 V100 고정 규약은 이번 로컬 실행에 적용하지 않는다. |
| D28 | **2026-09-22 smoke 조건.** 고정 TT prior `xrh4wk4l/040.pt` + RoboToken bank 128/128 · reference token 전부 mask · 팔 actuator 제외 · `designs=4`, `rounds=2`, `theta_steps=4096`, `num_envs=32`, `horizon=128`, `eval_seeds=3`, `seed=0`. |
| D29 | **2026-09-22 reward/종료 구현.** 매 step `Δ(world_x)/1.0 m` 누적 · 후진 음수 · 종료 step 이후 보상 0 · `calf_motor`/`gripper_collider` floor 접촉 또는 `up_z < 0.1` 즉시 종료 · 1.0 m 무낙상 도달만 success · 별도 alive/성공 보너스/추종/관절 penalty 없음. |
| D30 | **2026-09-22 학습 목표.** 1차 목표는 2 round 동안 형상 `E`와 보행 정책 `π_θ`가 함께 변하고, 동일 보상으로 `π_φ`가 갱신되는지 검증하는 것. 성공률 향상이나 최적 형상 발견은 이번 smoke의 결론으로 주장하지 않으며 `return`, `success`, `fell`, `theta_parameter_delta`, `phi_parameter_delta`를 기록한다. |
| D31 | **2026-09-22 본 학습 시작.** smoke 배선 통과 후 `rounds=20`, `designs=8`, `theta_steps=131072`, `num_envs=128`, `horizon=1000`, `eval_seeds=9`, `theta_lr=1e-4`, `goal=1.0 m`, `upright=0.1`, `action_scale=0.25`, `seed=0`으로 실행한다. OOM 또는 불안정 시 round checkpoint를 보존하고 `num_envs`만 단계적으로 낮춘다. |
| D32 | **2026-09-22 live viewer.** 학습은 headless로 유지하고 `watch_coevolution_viewer.py`가 `round_XXXX/theta.pkl`을 감시해 최신 best 형상을 MuJoCo GUI에서 π_θ로 재생한다. viewer는 평가·시각화 전용이며 학습 상태를 변경하지 않는다. |
| D33 | **2026-09-25 1 단계 완료.** 고정 TT prior + UMI gripper tracking 조건에서 RL guidance 로 morphology 생성과 motion tracking 동작을 확인했다. 이는 RL guidance 치환의 성립 검증이며 task-result co-design 검증은 아니다. |
| D34 | **2 단계는 공진화가 아니라 pi_theta 고정 실험으로 재설계.** D12·D26 공진화는 후속 3 단계로 보류하고, 먼저 pi_phi 단독 효과를 분리해 검증한다. |
| D35 | **형상/시뮬레이터는 TT 그대로 유지.** A1 URDF 로 바꾸지 않고 `umi_on_legs_plus_plus` design space + TT RoboToken + MJX 경로를 사용한다. |
| D36 | **locomotion reward 는 legged_robot 원본을 그대로 사용.** reward 항·scale·termination 설정을 임의로 빼거나 새로 추가하지 않고 원본 config 를 이식한다. |
| D37 | **A1/legged_gym 은 locomotion 설계 참고용.** URDF·robot dynamics 는 가져오지 않고 reward·observation·command 기반 PPO 구조를 참고한다. |
| D38 | **pi_theta 는 morphology-conditioned locomotion policy 로 선학습 후 동결.** TT design space 의 다양한 유효 morphology 를 사용해 충분히 학습하고 Stage 2 hardware search 중에는 weight 를 바꾸지 않는다. |
| D39 | **pi_phi reward = 동결 pi_theta 로 얻은 physics rollout episode return.** 동일 morphology 를 여러 seed 로 평가한 평균 return 을 hardware-generation terminal reward 로 사용한다. |
| D40 | **2 단계에서는 single-task flat locomotion 만 사용.** multi-task·계단·rough terrain·jump 는 single-task hardware generation 검증 후로 보류한다. |
| D41 | **surrogate·topology 확장·Sim2Real 은 현재 범위 밖.** simulator return 을 직접 사용하고 broad topology 는 Stage 2/3 성립 후 확장한다. |
| D42 | **repo 는 단계별로 분리한다.** 기존 coevolution 코드는 삭제하지 않고 archive/후속 Stage 3 로 보류하며 Stage 2 전용 locomotion/controller/hardware-guidance 경로를 새로 둔다. |

핵심 질문: *reference motion·reward gradient 없이, task 수행 결과만으로 diffusion 의 denoising
전략을 RL 로 학습할 수 있는가?*

---

## 2. 열린 안건

| Q | 안건 | 상태 |
|---|------|------|
| **Q6** | multi-task reward 합성식 G, physical penalty, task 별 제한 시간 | **Stage 3 이후로 보류** |
| **Q8** | topology 확장 범위(다리 수·DoF·연결 구조) | **Stage 2/3 성립 후 보류** |
| **Q9** | Stage 2 평지 locomotion command 를 고정 target 속도로 둘지 원본 command range 를 사용할지 | 미정 |
| **Q10** | morphology-conditioned pi_theta 선학습에 사용할 TT morphology sampling 수·분포 | 미정 |
| **Q11** | Stage 2 pi_phi 갱신 시 morphology 당 rollout seed 수와 batch 크기 | 미정 |

---

## 3. 근거 메모

**3.0 정책 고정의 편향 — 왜 D12 가 필수인가 (2026-09-21 사용자 지적)**
우리가 원하는 형상 점수는 `R(E) = max_θ J(E,θ)` 인데 π_θ 를 고정하면 `J(E,θ_고정)` 만 잰다 → **잠재력 높은 형상이 과소평가**된다.
그대로 π_φ 를 학습시키면 "본질적으로 좋은 형상" 이 아니라 **"지금 정책이 다루기 편한 형상"** 으로 수렴한다 (lock-in).
정책 고정 비교는 1 단에만 해당한다 (D24). 2 단은 D12 에 따라 형상과 정책을 번갈아 학습한다.

**3.1 왜 gradient 가 아니라 RL 인가**
TT 의 `∇_E R` 은 reward 가 트랜스포머 예측 토큰의 해석적 함수라서 가능하다 (`diffusion_guidance.py:291`,
물리 시뮬이 루프 밖). D9 로 reward 를 실제 rollout 성패로 바꾼 순간 그 gradient 는 존재하지 않는다.
RL 은 취향이 아니라 미분 불가능한 종단 보상에 credit 을 배분할 유일한 수단이다.

**3.2 Transform2Act — 빌려오는 것은 credit assignment**
T2A 는 [설계 step = reward 0] + [그 몸으로 수행 = 여기서만 reward] 로 설계 가치를 결과로 자동 정의한다.
대응: 설계 step ↔ denoising 마지막 10 step, 실행 ↔ π_θ rollout, 보상 ↔ 도달 AND 무낙상.
**단 한 MDP 로 잇지 않는다** — γ=0.99 로 1050 step 이으면 신호가 3e-5 로 소멸하므로 rollout 은 채점기로만.

**3.3 T2A 대비 우리 강점**
T2A 는 관절을 직접 add/remove 해 물리적으로 말이 안 되는 형상이 대량 나오고 공간이 조합 폭발한다.
우리는 denoiser 고정 + residual 이라 **생성 형상이 항상 prior manifold 위**에 있다 (D11).
이것이 D8(기존 design space 유지)의 실질적 값어치다.

**3.4 DiffuseBot / π_φ 학습 가능성**
DiffuseBot 의 embedding optimization·online data augmentation (상위 k% 로 denoiser fine-tune) = L2 로 채택.
π_φ 학습 조건 3 가지: ① 마지막 몇 step 에만 guidance (credit path 단축) ② 1 차는 contextual bandit
③ **DDIM eta = 1.0 필수** (eta=0 은 결정적이라 log-prob·exploration 없음).

**3.5 task set 세부 (D15)**
T1 보행(plane) / T2 상승(box) / T3 하강(box) / T4 불균형(hfield) / T5 점프(box), 전부 sphere-지형 충돌 ✅.
5 개 모두 "제한 시간 내 도달 AND 무낙상" 동일 틀 → G 는 성공률 벡터의 함수.
T5 는 모터 고정이라 성공이 순전히 형상에 달려 신호가 가장 강하고, prior 밖으로 밀 위험도 가장 크다.

---

**3.6 왜 2 단계를 다시 단순화했는가 (2026-09-25)**
1 단계에서 RL guidance 자체는 이미 동작했으므로, 이제 pi_phi 와 pi_theta 를 동시에 바꾸면 형상 효과와 제어기 효과가 다시 섞인다.
따라서 2 단계는 pi_theta 를 먼저 학습·고정하고 `J(E, theta_fixed)` 기준으로 pi_phi 의 task-result hardware generation 만 검증한다.

**3.7 왜 A1 URDF 를 쓰지 않는가**
TT prior 는 TT RoboToken/design space 의 morphology 분포를 학습했으므로 A1 URDF 로 교체하면 생성 prior 와 simulator model 사이에 새 변수가 생긴다.
A1/legged_gym 에서는 검증된 locomotion reward·observation·command 설계만 참고하고, 형상은 TT design space 를 유지한다.

**3.8 왜 legged_robot reward 를 그대로 쓰는가**
현재 목적은 reward engineering 이 아니라 task-result RL guidance 검증이므로 reward 항을 임의로 조정하지 않는다.
원본 reward config 를 그대로 사용해 locomotion evaluator 자체의 불확실성을 줄이고, episode return 을 pi_phi 의 terminal reward 로 사용한다.

---

## 4. 실험 규약

> §4.1~4.3은 1단 비교 실험의 당시 규약이다. 현재 2단 공진화 조건은 D26~D31과 `runs/coevolution_live_20260922/config.json`을 따른다.
> **조건이 적히지 않은 결과는 기록으로 쓰지 않는다.**

### 4.1 이전 V100 실험 환경 (2026-09-20 기록; 현재 공진화에는 D27·D31 적용)

| 항목 | 값 | 확인 |
|------|-----|------|
| GPU | Tesla V100-PCIE-32GB (sm_70) 1 장, `CUDA_VISIBLE_DEVICES=0` | `nvidia-smi -L` |
| 드라이버 / CUDA | 575.64.03 / 12.9. **CUDA 13 금지** (Volta 지원 제거) | `nvidia-smi` |
| 코드 | 실체 `RL/transformer-transformer`, 색인 `RL/tt_codesign/code/` | — |
| 파이썬 env | `.venv` (uv, py3.12). isaacgym conda env 와 완전 분리 | `.venv/bin/python -V` |
| 패키지 | jax 0.10.2 / mujoco·mjx 3.10.0 / torch 2.14.0+cu126 / brax PPO | `uv pip list` |
| design space | TT `umi_on_legs_plus_plus` (4 족+팔). **팔 고정**, 13 개 연속 uniform | `t2/robogen/umi_on_legs_plus_plus.py` |
| V100 필수 플래그 | `use_flash_attn=false` (sm_80+ 전용) / `use_mixed_precision=false` (bf16 없음) / `use_torch_compile=false` | — |
| 산출물 | `tt_codesign/runs/` · `results/` · `assets/` | — |
| 로깅 | `WANDB_MODE=offline`, 콘솔 `RL/logs/tmux_run/<세션>_<시각>.log` | — |
| 실행 | 반드시 tmux (`bash scripts/tmux_run.sh`), nohup 금지 | `CLAUDE.md` |

### 4.2 실험별 명시 항목 — 1 단 동등성 (논문 §3 프로토콜 그대로)

| 항목 | 값 |
|------|-----|
| design space | quadruped (이산 7 비트 = 128 조합 + 연속). 생성은 이 안에서만 (§5.10) |
| 궤적 | UMI 20-skill **held-out** (`data/july25th2025-huy-20skills-test.pkl`) |
| reward | `Σ exp(−‖ε_p‖²/0.01 − ‖ε_o‖²/0.5)` — **TT·RL 공통** |
| 평가기 | **choice 별 RL expert** (`rl_policies.zip`) — 양쪽 동일 |
| 종료 | 낙상 **또는** ε_p > **0.8 m** (quadruped 기준) |
| 지표 | **Pos Err (cm) · Orn Err (deg) · Survival (%) · Reward (에피소드 합) · 최적화 시간 (s)** |
| seed | **9** + 95 % CI |
| 비교 표 | 논문 Table 4 (Random / CMA-ES / Zeroth / DGS) 에 **RL 행 추가** |

> **1 단 task 는 논문 그대로 UMI 궤적이다.** 2 단은 D26의 전진 보행 task를 사용한다.
> 1 단의 목적이 Table 4 재현이므로 궤적을 바꾸면 비교할 공개 기준선이 사라진다.

### 4.3 비교 설계 — TT vs RL (D17, 2026-09-21)

L0~L3 사다리는 폐기. **형상 생성 방식만 다른 둘**을 같은 평가기·같은 지표로 비교한다.

| 조건 | 형상 생성 (guidance) | 비고 |
|------|---------------------|------|
| **TT** | TT prior + `∇_E R` DSG (`evaluate_hardware_opt_rl.py`) | 논문 재현 경로 |
| **RL** | π_φ residual guidance (D11) — 실제 rollout 결과가 reward | 우리 기여 |

**1단 평가는 choice별 RL expert와 동일한 9 seed·held-out UMI 궤적으로 한다** (D18). 지표는 Pos Err·Orn Err·Survival·Reward·시간 (D19).
§5.16의 전진 거리·기립도 진단은 당시 예비 실험이며, 현재 2단의 보행 reward와 구분한다.

---

### 4.4 현재 Stage 2 실험 규약 (2026-09-25)

| 항목 | 값 |
|------|-----|
| 형상 prior / design space | **TT `umi_on_legs_plus_plus` 그대로**, RoboToken → MJX model 경로 유지 |
| task | **single-task flat locomotion** |
| reward | **legged_robot 원본 reward config 전체를 그대로 이식**; 임의 항 삭제·추가 없음 |
| controller | `pi_theta(o,E,command)` morphology-conditioned PPO, Stage 2 전에 선학습 후 **동결** |
| hardware guidance | 고정 TT denoiser + residual `pi_phi`; 기존 Stage 1 RL guidance 경로 재사용 |
| simulator | **MJX/MuJoCo 유지**; Isaac Gym/A1 URDF 로 교체하지 않음 |
| evaluation | 최종 morphology 에서만 physics rollout, 여러 seed 평균 episode return 사용 |
| update | Stage 2 동안 **pi_phi 만 업데이트**, pi_theta 는 고정 |
| 비교 | unguided TT prior / RL-guided hardware generation; DSG 비교는 task reward 미분 가능성에 따라 별도 판단 |
| 제외 | coevolution, multi-task, topology 확장, surrogate, Sim2Real |

**Stage 2 완료 조건:** 동결된 동일 pi_theta 로 평가했을 때 RL-guided generation 이 학습 전/unguided 대비 더 높은 locomotion return 의 morphology 분포를 만든다.
이 단계의 성능은 `J(E, theta_fixed)` 이며 최종 `max_theta J(E,theta)` 주장으로 해석하지 않는다.

---

## 5. 실험 결과

> 읽는 순서 = 시간순. §5.1~5.8 = 1 차 준비 · §5.9~5.13 = TT 재현 · §5.14~5.16 = π_φ 시도
> · §5.17~5.19 = reward 진단 · §5.20~5.22 = 판정과 폐기.

**5.1 π_θ 학습 try_1~5 — 전부 폐기 (2026-09-20)**
try_1 PPO 발산 / try_2 정지 hack / try_3 계단 앞 정지 / try_4 terrain curriculum 오진 / try_5 0.029 m/s 퇴화보행.
원인은 내가 quadruped ctrl 평가를 `umi_on_legs` env 로 돌려놓고 "정책이 실패한다"고 오판한 것 (§5.2).
try_5 ckpt 126 M 만 fallback 보존. 진단 상세는 `runs/pi_theta_try*` 와 압축 로그.


**5.2 TT ctrl 정책은 쓸 수 있다 — env 오배선이 원인이었다 (2026-09-20)**
`d7lra7j0` 는 `umi_on_legs_plus_plus` 로 학습됐는데 평가 config 가 `umi_on_legs` 를 쓰고 있었다.
env 교정 후 `pos_err` 0.338 → **0.035**, `orn_err` 0.967 → 0.180, 완주 50 % → **100 %**.
합성 직선 궤적도 동작: 0.05 m/s 에서 pos_err 0.051 / 0.20 m/s 에서 0.346 → **속도 한계가 그 사이**.


**5.3 계단은 못 오른다 — 그래서 1 차 task 는 T1 보행 (2026-09-20)**
계단 3 종 명령 모두 x ≈ 0.37~0.45 에서 막히고 base z 가 한 번도 오르지 않는다 (넘어지는 게 아니라 막힘).
평지에서도 최대 도달점이 0.44~0.51 로 비슷 → 계단이 벽이 아니라 **지속 보행 능력 자체가 1 m 안팎**.
조기 종료는 전부 `pos_err > 0.8` (낙상 아님) — 목표가 로봇을 앞질러 끊긴다.


**5.4 처리량 — surrogate 불필요 (2026-09-20)**
병목은 물리가 아니라 트랜스포머다 (배치 64 에서 1114 ms vs 물리 84 ms = 7 %) → 단일 프로세스로 충분.
denoising 5 → 2 회로 줄이면 2.5 배 공짜, 품질 동일 (이동 1.027 → 1.025 m). 1 회는 36 step 만에 붕괴.
실측 **49 설계-step/s = 설계당 12.3 s** (`t2/eval/batched_ctrl.py`).


**5.5 목표 거리 0.8 m 확정 (2026-09-20, 설계 32 개 예비)**
이동거리 평균 0.687 / 표준편차 0.416 / 5 개는 전혀 못 움직임 → 설계 변별이 실재한다.
이진 판정 변별력 `4p(1−p)`: 0.5 m 0.61 / **0.8 m 1.00 (최대)** / 1.0 m 0.75 / 1.2 m 0.44.
→ 목표 0.8 m. §4.3 이 L0 에 요구한 "변별되는 난이도"가 이 값이다.


**5.6 기구학 — 계단 못 오른 건 물리 한계가 아니다 (2026-09-20)**
선 자세 몸통높이 0.237 m, 발 최대 들기 0.635 m, 단높이 0.10 m 통과 가능 설계 **100 %** (200 개).
즉 exploration barrier: 접근은 progress reward 가 꾸준히 늘어 쉽고, 첫 단은 협응 필요 + 낙상 위험.
**계단 앞에 서 있는 것이 국소 최적**이었다.


**5.7 D8 이탈은 철회됐다 — TT prior 를 그대로 쓴다 (2026-09-20 이탈 → 2026-09-21 철회)**
이탈 사유였던 "동결 π_θ 가 이산 조합 1 개만 제어 가능" 은 **expert 128 개로 해소**된다 (§5.10).
따라서 13 차원 자체 DDPM 은 폐기하고, **TT 사전학습 prior + `evaluate_hardware_opt_rl.py` 경로를 쓴다** (D8·D21).
자체 DDPM 으로 돌린 L2·L3 결과(§5.8)는 이 이유로도 인용하지 않는다.


**5.8 L0~L3 본 실행 (2026-09-20, tmux `L_walk`, 3 h 41 m)**
성공률 L0 0.480 / L1 0.820 / L2 0.895 / L3 **0.594**, 낙상 0.234 / 0.164 / 0.086 / 0.238.
표면상 판정은 §6 분기의 `L2 > L3` = "RL 없이 prior 재학습이면 충분, L3 재검토".
**단 이 수치는 §5.21 로 폐기 대상이다.** 원자료 `results/design_search_walk.jsonl`.


**5.9 TT 의 RL expert 정책은 받을 수 있다 — remind 이전 기술이 틀렸다 (2026-09-21)**
`rl_policies.zip` (real.stanford.edu/transformer-transformer/) 에 **128 개 per-choice PPO expert + `choice_to_ckpt_path.json`
+ options zarr + RoboToken bank** 가 전부 들어 있다 (`docs/starter.md:126`, `docs/data_generation.md:155`). 학습 불필요.
당시 미다운로드 상태였으나 이후 받아 §5.13 TT 행 재현에 사용했다. `scripts/evaluate_hardware_opt_rl.py` 가 정식 quadruped 평가 경로다.


**5.10 논문 확인 — D8 이탈의 전제가 무너졌다 (2026-09-21, arXiv 2607.25798)**
§2.4: quadruped design space 는 **이산 7 비트 = 128 조합 + 연속**으로 고정. 생성은 그 안에서만 일어난다.
**`rl_policies.zip` 의 expert 128 개가 그 128 조합 하나당 1 개**이므로 전 조합 평가 가능 → **TT prior 를 그대로 쓸 수 있다**.
§3: 평가기 = RL expert, 지표 = Pos Err·Orn Err·Survival·Reward, seed 9, 종료 = 낙상 또는 ε_p > 0.8 m. Table 4 가 기준선.


**5.11 다운로드 자산 감사 (2026-09-21 당시 상태)**
공개 아카이브 4 종 중 당시 `checkpoints.zip`(7.3 GB) · `data.zip`(288 MB) 보유, `rl_policies.zip`(9.3 GB) 다운로드 중이었다. 이후 §5.13에서 expert를 사용했다.
`blender_templates.zip` 은 시각화 전용이라 불필요. HF 의 bimanual dishwashing zarr 은 ctrl 재학습용이라 불필요.
디스크 여유 2.9 TB 로 제약 없음.


**5.12 우리 pos_err 은 논문의 5~13 배 (2026-09-21)**
논문 Table 4 quadruped: Random 3.4 cm / CMA-ES 1.9 / Zeroth·DGS 2.6, Survival 96~100 %.
우리 측정은 **12~34 cm** (§5.22·§5.16). 증류 ctrl + 자체 합성 보행 궤적을 쓴 탓으로 보인다.
**expert + UMI held-out 궤적으로 바꾸면 이 격차부터 사라져야 정상.** 안 사라지면 배선 문제다.


**5.13 TT 행 재현 성공 — 배선이 맞다 (2026-09-21, 궤적 20 × seed 9 = 180, 55 분)**
Quadruped/Tracking Only/Single/DSG: Pos Err **3.0 cm** (논문 2.6) · Orn Err **8.0°** (7.4) · **Survival 99.4 %** (99.4, 일치) · Reward 280.5.
명령 = `tt_codesign/code/repro_table4_TT.sh 16 guided_diffusion` (ckpt `xrh4wk4l/040.pt`, scale 100, eta 1.0, V100 플래그).
§5.12 의 10 배 격차가 사라졌다 → 포복·전복은 정책 결함이 아니라 **우리 평가 세팅의 산물**이었다.


**5.14 π_φ try_1 (REINFORCE) 실패 — action 차원 (2026-09-21, 30 업데이트에서 중단)**
action 차원 = step 당 6,694 (dyna_joint 1798 + fixed_joint 1900 + link 2730 + actuator 266) × 10 step = **66,940**.
log-prob 합 하나로 credit 을 배분하므로 표본 16 개로는 신호가 분산에 묻힌다 — reward 6.524 → 6.524, loss 부호 무작위.
교훈: 이 차원수에서는 REINFORCE 가 아니라 `∇_a Q` (critic) 가 필요하다 → try_2.


**5.15 π_φ try_2 (critic) 실패 — 문제는 방법이 아니라 실험 지점이다 (2026-09-21, 400 업데이트 45 분)**
Q_ψ(s,a) 17.6 만 파라미터 + `−Q(s,π(s))` 로 바꿨으나 reward 6.52 → 6.43, **Qloss 0.92** (표준화 타깃 분산의 8 %만 설명).
원인: 논문 Table 4 에서 **Zeroth = DGS (2.6 cm / 7.4° / 99.4 % 동일)** 이고 §3 이 "표본이 많으면 둘은 구분 불가" 라고 명시.
**n=16 quadruped single 은 guidance 효과가 0 인 지점**이라 학습할 신호 자체가 없었다. 프로토콜 선정 오류 (내 책임).


**5.16 rollout 분산이 설계 차이를 덮는다 — seed 1 은 측정 불가 (2026-09-21, 같은 설계 8 회)**
`torch.manual_seed` 가 어디에도 없어 정책 샘플링 난수가 자유이고, 배치 구성이 바뀌면 같은 설계도 다른 rollout 이 된다.
prior_2 × 8: 전진 1.183 ± 0.386 (0.79~2.04), 기립도 +0.355 ± 0.256 — **깨끗한 보행은 8 회 중 1 회뿐** (설계 속성 아님).
L3_best × 8: 전진 1.139 ± 0.281 로 **거리는 겹치지만**, 기립도 −0.495 ± 0.287 / pos_err 0.273 ± 0.067 은 **겹치지 않는다**.


**5.17 TT 의 reward 는 두 개다 — 내가 잘못된 쪽을 D22 로 정했다 (2026-09-21, 개정)**
**① 제어 학습용** (`mjx_umi_on_legs.yaml`, 10 항) = expert 만들 때만. **② 설계 평가용** (`config/reward_fns/*.yaml`, 1~2 항) = Table 4·Fig 5·DSG 의 `∇R̂` 전부.
논문이 형상에 쓰는 건 **②** 이고 (§1 "user-defined reward functions", §3 "unseen reward functions zero-shot"), ①은 형상 점수에 쓰인 적이 없다.
내가 D22 에 ①을 적어 `joint_acc` 가 보행을 벌하게 만들었다 (§5.19). D22 는 결과 기반 reward 하나로 개정.


**5.18 TT ① 제어 학습 reward 의 내용 (참고용 — 형상 점수에는 쓰지 않는다)**
양의 항은 `exp_pose_err` **+4.0** = exp(−orn²/0.5 − pos²/0.01) 하나뿐이고 나머지는 전부 penalty.
**우리가 빠뜨린 보호막이 여기 다 있다**: `penalty_orientation` −1.0(기립), `penalty_bad_contact` −1.0(몸통·모터 접지),
`termination_fellover_threshold` **0.1** (우리 평가 env 는 0.5 로 훨씬 느슨), `penalty_termination` −10.0.

**5.19 D22(구) 는 보행 task 를 벌한다 — 계수가 task 의존이다 (2026-09-21, 같은 형상 2 개)**
같은 채점기로 task 만 바꾸니 `joint_acc` 가 **8~19 배**로 뛰고 합계 부호가 뒤집혔다 (조작 +428/+685 → 보행 −6544/−12060).
발이 매 step 바닥을 치기 때문이고, 솔버 탓이 아니다 (MJX 값으로 바꿔도 −6909 → −7940 로 더 커짐).
부수 확인: **순위도 뒤집힌다** — 보행은 TT 형상, 조작은 RL 형상이 우세. task 가 바뀌면 좋은 설계가 바뀐다.


**5.20 RL 행 완주 — 생성은 되고 수치도 준한다 (2026-09-21, 궤적 20 × seed 9, TT 행과 동일 프로토콜)**
Pos Err **3.0 cm** (TT 3.0) · Orn Err **8.6°** (TT 8.0) · Reward **277.8** (TT 280.5) · **Survival 97.8 %** (TT 99.4).
즉 **π_φ 를 DSG 자리에 끼워도 파이프라인이 깨지지 않고 TT 급 수치가 나온다** — 치환의 기계적 성립은 확인.
단 π_φ 는 §5.15 대로 학습이 안 된 상태(action 이 상한 0.02 에 포화)라, Survival 1.6 %p 하락(조기종료 0.6→2.2 %)은 그 잡음 비용으로 본다.


**5.21 성공 판정 결함 — 1 차 결과 폐기 (2026-09-21)**
① `design_search_walk.py:48` 이 `success = dist >= goal` 로 **명세의 "AND 무낙상" 을 빠뜨렸다**.
② `base_env.py:346` 의 낙상 판정은 free geom z 축만 봐서 **배 깔고 수평이면 안 걸린다** (코드에 TODO 기재).
③ 결과: 거리 ↔ 기립도 **−0.535**, 거리 ↔ 몸통접지율 **+0.551** — 누울수록 멀리 간다.


**5.22 자세 감사 — 측정이 틀렸다 (2026-09-21, 설계 17 개)**
리셋 시 기립도 1.000 으로 지표 검증 통과. 17 개 중 정상 보행은 2 개뿐, **설계공간 중앙(center)조차 기립도 0.122**.
거리 0.8 m 통과 14 개 중 10 개가 기립도 < 0.3. `prior_2` 가 0.987 로 유일하게 깨끗했으나 §5.16 이 이를 뒤집는다.
반증: "EE 추종이라 기어가도 된다" 는 틀렸다 — pos_err ↔ 접지율 **+0.790** (포복은 정책의 실패다). `results/posture_audit.json`.

**5.23 2 단 reward 실측 — 변별되고, 병목은 종아리 모터다 (2026-09-21, 목표 1.0 m)**
TT 형상 score **0.15** (298 step 종료) vs RL 형상 **0.38** (453 step) — 2.5 배 변별. 둘 다 1.0 m 미달이라 성공률은 아직 0.
종료 원인은 **둘 다 `left_leg/calf_motor` 바닥 접촉**이고 `upright_mean` 0.996·0.978 = **넘어진 게 아니라 종아리 모터가 긁혔다**.
즉 reward 가 가리키는 건 자세 붕괴가 아니라 **설계 결함**(모터 장착 높이·종아리 길이) — co-design 이 고쳐야 할 바로 그것.

**5.24 낙상 종료 링크 목록은 실측으로 정했다 (2026-09-21, `contact_audit.py`)**
"발 아닌 전부" 로 잡으면 정상 보행도 죽는다 — TT 형상은 calf·rod_insert 가 2 %, RL 형상은 link4 10 %·calf 7 % 스친다.
그래서 **TT 의 `penalty_bad_contact_pairs` 목록(calf_motor ×4 + gripper_collider)을 그대로** 쓴다. 우리가 임의로 정하지 않는다.
웹 검증: legged_gym 은 지정 링크 접촉력 > 1 N 즉시 reset, co-design 문헌도 "upright 인 개체만 평가" — 셋 다 **누적 지표가 아니라 이진 종료**.

**5.25 2단 공진화 실행 조건 기록 (2026-09-22)**
이번 1차 smoke는 RTX 4070 SUPER에서 `rounds=2`, `designs=4`, `theta_steps=4096`, `num_envs=32`, `horizon=128`, `eval_seeds=3`으로 시작한다.
목표는 성능 결론이 아니라 reference-free 형상 생성 → 공유 π_θ PPO → 동일 reward 평가 → π_φ 갱신의 한 round 연결과 checkpoint 저장 확인이다.
판정값은 `return_mean`, `success_mean`, `fell`, `theta_parameter_delta`, `phi_parameter_delta`이며 성공률 향상이나 최적 형상 발견은 후속 본 학습에서 판단한다.

**5.26 2단 smoke 완료 — 배선 통과, 보행 성공 미확인 (2026-09-22)**
Round 0/1 모두 유효 형상 4개·invalid 0개·checkpoint 저장·π_φ 갱신에 성공했고, Round 1에서 `theta_parameter_delta=0.4165`, `phi_parameter_delta=0.0517`로 두 정책이 실제 갱신됐다.
하지만 `success_mean=0.0`이고 `return_mean`은 **0.0124 → -0.0553**으로 악화됐다. 이는 보행 성능 향상의 증거가 아니며, 이후 D31 조건으로 본 학습을 시작했다.
산출물은 `runs/coevolution_smoke/round_0000~0001/{theta.pkl,phi.pt,best.xml,designs.json}` 및 `history.jsonl`이다.

**5.27 본 학습 기록 — 완료 round 0~2 (2026-09-22)**
`runs/coevolution_live_20260922/history.jsonl` 기준 8개 형상씩 유효·invalid 0, `return_mean` **0.0063 → 0.0293 → 0.0618**, `success_mean` **0 → 0 → 0**이다.
매 round π_φ가 갱신됐고 round 1·2의 `theta_parameter_delta`는 2.4609·1.9075다. return 상승은 관찰됐으나 아직 1.0 m 성공이나 최적 형상 발견의 근거는 아니다.
기록 확인 시점의 마지막 저장 checkpoint는 `round_0002`; `train.log`에는 round 3의 controller 8/8까지 기록됐다. round 3은 완료 기록이 없어 성능 판정에서 제외한다.


---

**5.28 Stage 1 완료 — RL guidance 치환 동작 확인 (2026-09-25)**
고정 TT prior + UMI gripper tracking 조건에서 RL guidance 로 morphology 를 생성하고 생성 형상에서 motion tracking 이 정상 동작하는 것을 확인했다.
따라서 이후 핵심은 guidance 가능성 재검증이 아니라 reference-motion-free locomotion return 으로 pi_phi 를 학습하는 Stage 2 이다.

**5.29 π_θ fine-tune 1 차 실패 — warm start 가 무너졌다 (2026-09-21, RL 형상 40 M step)**
출고 expert(pos_err **0.323** / orn_err 0.563) 에서 출발했는데 **40 M 내내 단 한 epoch 도 baseline 을 못 넘었다**
(최종 0.594 / 2.529). ep_len 이 1754 ↔ 28 ↔ 2000 ↔ 20 으로 100 배 진동하고, **짧게 죽은 epoch 일수록 pos_err 이 낮다**
— 낮은 pos_err 은 추종이 아니라 조기종료의 산물이다. orn_err 은 전 구간 미회복 = 이쪽이 진짜 신호.
기록·체크포인트 `runs/pi_theta_RL4/`. 의심 순위 ① fine-tune LR 이 출고 학습값 그대로 ② D22 종료(bad_contact+fellover 0.1)가
warm start 직후 상시 발동. **40 M 은 렌더 단계 X11 끊김으로 프로세스가 죽었을 뿐 학습은 완주**했다 (→ `MUJOCO_GL=egl` 고정).

---

## 6. 결정 로그

| 날짜 | 내용 |
|------|------|
| (이전) | D1~D7 — 변천사 문서 §1~§17 |
| 2026-09-19 | **D8** design space 고정 / **D9** 결과 기반 task / **D10** 성공 판정·팔 고정 / **D11** residual guidance |
| 2026-09-20 | **D12** 공진화 / **D13** MJX (Isaac 은 RT core 필요 → V100 불가) / **D14** 실물 제외 / **D15** task 5 종 |
| 2026-09-20 | §4 실험 규약 신설, L0~L3 사다리 설계. D8 이탈 승인 (§5.7) |
| 2026-09-20 | **D16** π_θ 학습 폐기 → TT ctrl 사용, 1 차 task 를 T2 계단 → **T1 보행** 으로 변경 |
| 2026-09-20 | L0~L3 본 실행 (§5.8) |
| 2026-09-21 | **1 차 결과 폐기 판정** (§5.21·§5.22·§5.16). Q10·Q11·Q12 신설. remind 서술 규칙 = 항목당 3 줄 |
| 2026-09-21 | **정정**: "PPO expert 128 개는 없다" 는 틀렸다. `rl_policies.zip` 으로 받을 수 있다 (§5.9) |
| 2026-09-21 | **D17** L0~L3 폐기 → **TT vs RL** 2 조건 비교. `rl_policies.zip` 9.3 GB 다운로드 착수 |
| 2026-09-21 | 논문 정독 → **D18~D21** 신설 (평가 프로토콜·지표·종료·TT 경로). **2 단 구조를 §0 으로 승격**. D8 이탈 철회 |
| 2026-09-22 | **D27~D30** 실제 로컬 GPU 환경·1차 공진화 조건·D22 reward 구현·학습 목표 확정. RoboToken bank 128/128 및 coevolution preflight 통과 |
| 2026-09-22 | smoke 2 round 완료 (§5.26), D31 조건으로 본 학습 시작; 완료된 round 0~2 결과 기록 (§5.27) |
| 2026-09-25 | **D33~D42 개정.** Stage 1 완료 확인; 공진화 우선 경로를 Stage 3 로 보류하고 Stage 2 를 `TT morphology + frozen locomotion pi_theta + legged_robot reward + pi_phi only` 로 단순화 |
| 2026-09-25 | **A1 역할 확정.** A1 URDF 는 사용하지 않고 legged_gym/legged_robot 의 locomotion reward·observation·PPO 설계만 참고; simulator 는 MJX 유지 |
| 2026-09-21 | π_θ fine-tune 1 차 40 M 실패 (§5.29). LR·종료조건 재검토 후 2 차. 학습은 **`MUJOCO_GL=egl` 필수** |

---

## TODO

### T1. Stage 1 기록 봉인 — 완료

**완료:** 고정 TT prior + UMI tracking + RL guidance 에서 morphology 생성과 motion tracking 동작을 확인했다.
이 경로는 회귀 검사용으로 보존하고 Stage 2 에서는 UMI trajectory/tracking reward 를 사용하지 않는다.

### T2. Repo 를 Stage 2 기준으로 정리

기존 `train_coevolution.py`, 과거 reward·공진화 스크립트는 삭제하지 말고 `archive/` 또는 Stage 3 경로로 이동한다.
Stage 2 전용 `locomotion_env`, `train_locomotion_policy`, `eval_locomotion_policy`, `train_hardware_guidance` 경로를 분리한다.

### T3. TT locomotion environment 작성

TT `umi_on_legs_plus_plus` morphology → MJX model 생성 경로는 그대로 사용하고 평지 single-task locomotion environment 를 만든다.
A1 URDF/Isaac Gym 으로 바꾸지 않으며 legged_robot 의 observation·command 구조만 필요한 범위에서 이식한다.

### T4. legged_robot 원본 reward 그대로 이식

`LeggedRobot` 원본 reward config 의 항·scale·termination 설정을 그대로 옮기고 임의로 reward 를 빼거나 새 항을 추가하지 않는다.
Stage 2 hardware reward 는 이 환경에서 얻은 episode return 자체로 정의한다.

### T5. Morphology-conditioned locomotion pi_theta 선학습

TT design space 에서 다양한 유효 morphology 를 샘플링해 `pi_theta(o,E,command)` 하나를 충분히 PPO 학습한다.
형상별 locomotion 성능·분산을 확인한 뒤 checkpoint 와 observation normalization 을 저장한다.

### T6. pi_theta 동결 및 evaluator 검증

Stage 2 시작 전에 pi_theta weight 와 normalization 을 고정하고 동일 morphology 를 여러 seed 로 반복 rollout 한다.
seed 간 분산보다 morphology 간 return 차이가 식별 가능한지 확인한 뒤 hardware search 로 넘어간다.

### T7. Stage 2 RL-guided hardware generation 연결

기존 Stage 1 의 고정 TT denoiser + residual pi_phi 경로를 재사용해 noise 에서 최종 TT morphology E 를 생성한다.
중간 denoising 에서는 physics 를 돌리지 않고 최종 E 만 동결 pi_theta 로 MJX rollout 한다.

### T8. Episode return 으로 pi_phi 학습

동일 E 의 여러-seed episode return 평균을 terminal reward 로 사용해 pi_phi 만 업데이트한다.
`Noise → RL guidance → E → frozen pi_theta rollout → return → pi_phi update` 를 반복한다.

### T9. Stage 2 효과 검증

unguided TT prior 와 RL-guided generation 을 동일 TT prior·controller·seed·sim budget 으로 비교한다.
평균/best return, 성공률, 낙상/termination, morphology 분포, optimization cost 를 기록한다.

### T10. Task-specific morphology 변화 분석

동일 noise seed 의 guidance 전후 형상과 학습 전후 morphology parameter 분포를 비교한다.
높은 return 과 반복적으로 연결되는 형상 특성이 있는지 확인해 RL guidance 가 단순 noise 가 아님을 검증한다.

### T11. Stage 2 통과 후에만 Hardware–Policy Co-design 재개

Stage 2 가 성립하면 과거 coevolution 코드를 복구해 pi_phi 와 pi_theta 를 분리된 timescale 로 함께 최적화한다.
이때 `J(E, theta_fixed)` 에서 `max_theta J(E,theta)` 를 향하는 Stage 3 로 해석한다.

### T12. Multi-task / topology / surrogate 는 후속

single-task Stage 2 → single-task co-design Stage 3 가 먼저 성립한 뒤 multi-task 를 단계적으로 추가한다.
broad topology, surrogate, Sim2Real 은 그 이후 확장으로 유지한다.

---

## 현재 우선순위

**T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9** 순서로 진행한다.
현재 핵심 목표는 **검증된 고정 locomotion policy 아래에서 실제 locomotion return 만으로 RL guidance 가 더 좋은 TT morphology 를 생성하는지 확인하는 것**이다.

### 현재 실험 흐름

`TT design space morphology sampling`
→ `morphology-conditioned pi_theta locomotion 학습`
→ `pi_theta freeze`
→ `Noise + TT prior + pi_phi`
→ `Final morphology E`
→ `MJX locomotion rollout`
→ `legged_robot episode return`
→ `pi_phi update`

### 이후 흐름

`Stage 2 성공`
→ `pi_phi + pi_theta co-design`
→ `multi-task`
→ `broad topology design space`
→ `One Hardware + One Multi-task Policy`
