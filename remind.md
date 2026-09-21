# TT 확장 연구 — 결정·실험 기록 (remind)

> **정본.** 결정은 §1 에 D 번호로, 실험은 §4 규약대로, 결과는 §5 에 넣는다.
> **서술 규칙 (2026-09-21 사용자 지시): 모든 항목은 결과부터 쓰고 3 줄 이내.** 긴 산문 금지.
>
> 위치 `RL/tt_codesign/remind.md` · 변천 원문 `RL/Transformer_Transformer_연구계획_파이프라인_변천사.md`
> · 원 논문 코드 https://github.com/seung-ddak/transformer-transformer · 기존 BO co-design `quad/daily.md`

---

## 0. 목적

TT 의 "noise → denoising → 형상" 틀은 유지하되, reference motion + reward gradient guidance 를
**실제 task 수행 결과를 reward 로 삼는 RL guidance** 로 교체한다. 산출물은 시뮬 상의 (E\*, π\*).

### 연구 구조 — 2 단 (처음부터 확정. 흐려지지 않게 여기 고정한다)

| 단 | 무엇을 하나 | 묻는 것 | 비교 상대 |
|----|------------|---------|-----------|
| **1 단 · 동등성** | **TT 와 RL 둘 다 TT 의 tracking reward** 로 형상 최적화 | RL guidance 가 DSG 와 **동등한가** | DSG (논문 Table 4) |
| **2 단 · 확장** | **reward 하나**(걷기+무낙상)로 π_φ 와 π_θ 를 **함께** 학습 (D12·D22) | DSG 가 원리적으로 못 하는 것을 하는가 | 없음 (순수 novelty) |

1 단이 성립해야 2 단이 의미를 갖는다. 1 단은 "이기는가"가 아니라 **"동등한가"** 를 묻는다 —
§3.1 대로 reward 를 실제 결과로 바꾸는 순간 `∇_E R` 은 존재 자체가 불가능하므로, RL 은 성능
향상 수단이 아니라 gradient 가 사라진 자리를 메우는 유일한 수단이다.

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
| D16 | **π_θ 를 학습하지 않는다.** TT 공개 ctrl `d7lra7j0/040.pt` 를 그대로 쓴다. 1 차 task 는 T1 보행 |
| D17 | **L0~L3 사다리 폐기 (2026-09-21).** 비교는 **TT** (논문 그대로) vs **RL** (우리 guidance) 둘만. §0 목적에 직결 |
| D18 | **평가 프로토콜 = 논문 §3 그대로.** 평가기 = choice 별 **RL expert**, seed **9** + 95 % CI, held-out UMI 궤적 |
| D19 | **1 단 측정 지표 = Pos Err · Orn Err · Survival · Reward · 최적화 시간.** 전진 거리는 쓰지 않는다 (§5.16) |
| D20 | **종료 조건 = 낙상 또는 ε_p > 0.8 m** (논문 quadruped 기준). 자체 낙상 판정식은 쓰지 않는다 |
| D21 | **"TT 그대로" = `evaluate_hardware_opt_rl.py` + `rl_policies.zip`** (TT prior + DSG + choice 별 expert) |
| D22 | **2 단 reward 확정 (2026-09-21)**: 낙상 = TT 의 `penalty_bad_contact_pairs` 링크 접촉 **또는** up_vec·z < 0.1 → 즉시 종료 / 점수 = 종료 전 전진 진척도 (목표 **1.0 m** 정규화) / 성공 = 종료 없이 도달. π_φ·π_θ 가 같은 이 reward 로 학습 |
| D23 | **"형상에 맞는 최적 정책" 은 저절로 생기지 않는다.** π_θ 를 D22 reward 로 같이 학습시킬 때만 생긴다 → D12 가 필수 |
| D24 | **π_θ 고정으로 잰 형상 점수는 하한선이다** (`J(E,θ_고정) ≤ max_θ J`). **1 단에만 해당** — 2 단은 D12 로 이 한계를 푼다 |
| D25 | **π_θ 는 형상 조건부 + expert warm start + 2-timescale.** 형상마다 PPO 를 새로 돌리지 않는다 (곱셈 폭발·닭과달걀 회피) |

핵심 질문: *reference motion·reward gradient 없이, task 수행 결과만으로 diffusion 의 denoising
전략을 RL 로 학습할 수 있는가?*

---

## 2. 열린 안건

| Q | 안건 | 상태 |
|---|------|------|
| **Q15** | 2 단 task = 전진 보행(§0)인가, UMI 궤적 유지인가 | 미정. 1 단은 논문 그대로 UMI 로 간다 (§4.2) |
| **Q6** | multi-task reward 합성식 G, physical penalty, task 별 제한 시간 | 미정 |
| **Q8** | 2 단계 scope (topology 확장) | 미정 (Q2 동반) |

---

## 3. 근거 메모

**3.0 정책 고정의 편향 — 왜 D12 가 필수인가 (2026-09-21 사용자 지적)**
우리가 원하는 형상 점수는 `R(E) = max_θ J(E,θ)` 인데 π_θ 를 고정하면 `J(E,θ_고정)` 만 잰다 → **잠재력 높은 형상이 과소평가**된다.
그대로 π_φ 를 학습시키면 "본질적으로 좋은 형상" 이 아니라 **"지금 정책이 다루기 편한 형상"** 으로 수렴한다 (lock-in).
1·2 단이 고정을 쓰는 건 실패 원인을 π_φ 로 분리하기 위한 것이고, **결과는 하한선으로만 읽는다** (D24). 완화 요인 = expert 가 이미 연속 변형 2 만 개를 DR 로 덮는다.

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

## 4. 실험 규약

> 모든 실험에 동일 적용. 결과 기록 시 4.1·4.2 를 복사해 바뀐 항목만 고친다.
> **조건이 적히지 않은 결과는 기록으로 쓰지 않는다.**

### 4.1 고정 환경 (2026-09-20 확정)

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

> **task 는 논문 그대로 UMI 궤적이다.** §0 이 말한 전진 보행은 2 단에서 다룬다 (Q15).
> 1 단의 목적이 Table 4 재현이므로 궤적을 바꾸면 비교할 공개 기준선이 사라진다.

### 4.3 비교 설계 — TT vs RL (D17, 2026-09-21)

L0~L3 사다리는 폐기. **형상 생성 방식만 다른 둘**을 같은 평가기·같은 지표로 비교한다.

| 조건 | 형상 생성 (guidance) | 비고 |
|------|---------------------|------|
| **TT** | TT prior + `∇_E R` DSG (`evaluate_hardware_opt_rl.py`) | 논문 재현 경로 |
| **RL** | π_φ residual guidance (D11) — 실제 rollout 결과가 reward | 우리 기여 |

**평가는 한 개의 동일 평가기로 한다** — 생성 파이프라인이 무엇을 썼든, 나온 형상 둘을
같은 제어정책·같은 시드 집합·같은 고정 카메라로 굴린다. 측정값은 거리가 아니라
**기립도·pos_err·무낙상** (§5.16: 거리만으로는 두 설계가 겹친다). 평가기 선택은 Q13.

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
**미다운로드 상태.** `scripts/evaluate_hardware_opt_rl.py` 가 이것으로 TT 정식 quadruped co-design 평가를 돌린다 → Q11 의 답.


**5.10 논문 확인 — D8 이탈의 전제가 무너졌다 (2026-09-21, arXiv 2607.25798)**
§2.4: quadruped design space 는 **이산 7 비트 = 128 조합 + 연속**으로 고정. 생성은 그 안에서만 일어난다.
**`rl_policies.zip` 의 expert 128 개가 그 128 조합 하나당 1 개**이므로 전 조합 평가 가능 → **TT prior 를 그대로 쓸 수 있다**.
§3: 평가기 = RL expert, 지표 = Pos Err·Orn Err·Survival·Reward, seed 9, 종료 = 낙상 또는 ε_p > 0.8 m. Table 4 가 기준선.


**5.11 다운로드 자산 감사 (2026-09-21)**
공개 아카이브 4 종 중 `checkpoints.zip`(7.3 GB) · `data.zip`(288 MB) 보유. **`rl_policies.zip`(9.3 GB) 다운로드 진행 중**.
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


**5.23 2 단 reward 실측 — 변별되고, 병목은 종아리 모터다 (2026-09-21, 목표 1.0 m)**
TT 형상 score **0.15** (298 step 종료) vs RL 형상 **0.38** (453 step) — 2.5 배 변별. 둘 다 1.0 m 미달이라 성공률은 아직 0.
종료 원인은 **둘 다 `left_leg/calf_motor` 바닥 접촉**이고 `upright_mean` 0.996·0.978 = **넘어진 게 아니라 종아리 모터가 긁혔다**.
즉 reward 가 가리키는 건 자세 붕괴가 아니라 **설계 결함**(모터 장착 높이·종아리 길이) — co-design 이 고쳐야 할 바로 그것.

**5.24 낙상 종료 링크 목록은 실측으로 정했다 (2026-09-21, `contact_audit.py`)**
"발 아닌 전부" 로 잡으면 정상 보행도 죽는다 — TT 형상은 calf·rod_insert 가 2 %, RL 형상은 link4 10 %·calf 7 % 스친다.
그래서 **TT 의 `penalty_bad_contact_pairs` 목록(calf_motor ×4 + gripper_collider)을 그대로** 쓴다. 우리가 임의로 정하지 않는다.
웹 검증: legged_gym 은 지정 링크 접촉력 > 1 N 즉시 reset, co-design 문헌도 "upright 인 개체만 평가" — 셋 다 **누적 지표가 아니라 이진 종료**.

**5.22 자세 감사 — 측정이 틀렸다 (2026-09-21, 설계 17 개)**
리셋 시 기립도 1.000 으로 지표 검증 통과. 17 개 중 정상 보행은 2 개뿐, **설계공간 중앙(center)조차 기립도 0.122**.
거리 0.8 m 통과 14 개 중 10 개가 기립도 < 0.3. `prior_2` 가 0.987 로 유일하게 깨끗했으나 §5.16 이 이를 뒤집는다.
반증: "EE 추종이라 기어가도 된다" 는 틀렸다 — pos_err ↔ 접지율 **+0.790** (포복은 정책의 실패다). `results/posture_audit.json`.


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

---

## 7. 다음 할 일

**1 단 (완료·미결)**
1. TT 행 재현 ✅ (§5.13) · RL 행 완주 ✅ (§5.20). 단 π_φ 가 학습 안 된 상태라 **치환의 기계적 성립까지만** 확인.
2. 미결: guidance 효과가 유의한 지점을 못 찾았다 (§5.15 n=16 효과 0, 쌍대 n=1 도 t=1.22). 1 단 동등성은 **판정 보류**.

**2 단 (진행 예정)**
3. **reward 하나 확정** — 걷기 + 무낙상, 결과 기반 (D22). 구체 임계값(도달 거리·기립도 τ·접지율 κ)은 Q14.
4. **π_θ 를 그 reward 로 학습** — 형상 조건부, expert warm start, 2-timescale (D12·D25).
   배선 완료: `extract_best_designs.py` → `enumerate_design_store.py` → `finetune_pi_theta.sh`.
5. **π_φ 를 같은 reward 로 학습** — critic 구조 그대로, 원천만 rollout 으로 교체 (`d22_reward.py` 틀 재사용).
6. 그 다음: Q6 (multi-task 합성식 G) → Q8 (topology 확장).
