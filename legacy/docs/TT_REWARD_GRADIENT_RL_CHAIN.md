# 공통 TT → Reward-gradient → RL 잔차 순차 실행 설계

상태: **실행기·원본 데이터 어댑터·원본 sampler 잔차 연결 구현. 연결 검증 후 순차 실행**.
2026-11-10 기준. 목표는 같은 dynamics prior에서 RL 잔차가 실제 과제 수행을
개선하는지 검증하는 것이다. TT 예측 보상 상승을 실제 성공으로 취급하지 않는다.

## 1. 비교의 정의

공통 TT θ는 수집한 실제 dynamics를 원본 diffusion 손실로 사전학습한다.
이후 다음 세 조건 모두 같은 EMA checkpoint·정규화·sampler를 사용한다.

| 조건 | TT θ | 별도 학습 | 생성 시 적용 |
|---|---|---|---|
| U: 무보정 | 고정 | 없음 | 원본 sampler |
| G: reward-gradient | 고정 | 없음 | 원본 reward-gradient guidance |
| R: RL 잔차 | 고정 | PPO 정책 φ·critic | 매 denoising 단계의 bounded residual |

따라서 G는 ‘reward gradient로 TT 가중치 학습’이 아니라 **학습한 TT를 이용한
reward-gradient 생성**이다. R도 TT 전체를 미세조정하지 않는다. 이 정의는 기존에
합의한 ‘동결 TT + RL 잔차’와 같다. TT 가중치까지 RL/보상으로 바꾸는 실험은 이번
비교에 섞지 않는다. G의 결과나 최종 latent를 R의 학습 초기값으로 넘기지 않는다.
두 조건은 자원 사용 순서만 이어지며, 공통 checkpoint에서 독립적으로 시작한다.

## 2. 자동 실행 순서와 통과 조건

| 순서 | 작업 | 다음 단계로 넘기는 산출물·조건 |
|---|---|---|
| 0 | 원본 데이터·형상·source hash·물리 설정 검사 | 검사 통과한 immutable input manifest |
| 1 | 원본 t2 토큰 변환·학습/검증 분리 | 토큰 ↔ 실제 모터 목표·상태 round-trip, split·정규화 |
| 2 | 공통 TT 사전학습 | 로드 가능한 raw/EMA checkpoint, 유한 loss, 원본 설정 계약 검사 |
| 3 | 무보정 U 및 G 개발 평가 | 같은 후보 잡음의 실제 물리 실행, guidance scale 확정 |
| 4 | PPO 잔차 연결 시험 | 0 잔차 재현·보상 재계산·정답 잔차 학습·짧은 물리 실행 검사 |
| 5 | R 정책 학습 | TT 해시 불변, 별도 개발 평가로 선택한 PPO checkpoint |
| 6 | 잠근 최종 평가 U/G/R | 같은 과제·잡음 목록·물리·후보 수의 전체 결과와 보고서 |

한 단계가 예외, NaN, 메모리 초과 또는 계약 위반으로 실패하면 **후속 단계는 시작하지
않는다**. 수치상 실패와 낮은 과제 성능을 구분한다. 실행 계약이 맞고 정상적으로
물리가 실행된다면 낮은 무보정 성능만으로 RL 실험을 차단하지 않는다.
학습 종료를 수렴이나 과제 성공으로 자동 판정하지 않는다.

구현할 실행기는 이 고정 순서를 subprocess로 하나씩 실행하고, 단계별
`protocol.json`, `status.json`, 로그, 입력/출력 해시, 종료 코드를 기록한다.
완료 산출물을 검증한 단계만 재개 시 건너뛴다. checkpoint·로그는 원자적으로
저장하고, 실패한 단계는 산출물을 지우지 않은 채 멈춘다. 이 문서는 실행기가 아니다.

## 3. 데이터와 원본 TT 계약

입력: `runs/local_mujoco_10k_20261109/`.

- 학습 후보: `rollouts/`의 200개 형상·40,000 episodes·10,000 source motions.
  최종 audit: 12,017,076 valid transitions, 25,588 falls. 낙상 전의 유효 전이는
  유지하며 padding·종료 뒤 전이는 손실에서 제외한다. 성공 자료로 재분류하지 않는다.
- `heldout/`의 20개 형상은 TT·정규화·PPO 학습에서 제외하고, 최종 물리 일반화
  평가에만 쓴다. 학습 원본의 모든 source hash를 seed=0으로 섞어 80/20 분리하고,
  모든 형상·시작 variant·겹치는 window에 같은 source 분할을 적용한다.
- 50Hz 제어, 200Hz physics, kp=30, torque ±33.5Nm, 실제 `target_q`, 지면·접촉·
  초기화·종료를 고정한다. 기존 30Hz 자료는 합치지 않는다.
- 원본 t2 토큰에는 하드웨어 구조, 링크 pose 등 현재 축약 42D state에 없는 항목이
  필요하다. 원본 MJCF/E와 저장된 qpos/qvel로 MuJoCo forward를 수행해 복원하고
  원본 adapter에 연결한다. 실제 궤적을 CALM 목표 궤적으로 대체하지 않는다.
- 압축 NPZ 전체를 GPU/RAM에 적재하지 않는다. 순차 변환 후 원본 데이터 로더가
  window 단위로 읽을 수 있게 저장하며, 정규화는 학습 split에서만 계산한다.

원본 기준은 이 컴퓨터의 `../transformer-transformer/`이며, upstream 원본과 로컬
수정본의 차이를 기록하고 실제 resolved config를 보존한다.

| 항목 | 이번 기준 |
|---|---|
| backbone | `t2.model.core.DiT`, small: width 256 / depth 8 / heads 4 / AdaLN |
| task | quadruped hardware diffusion addon, 원본 modality adapters·conditioning·MSE |
| diffusion | hardware_gen의 100 train timesteps 및 원본 inference schedule |
| prediction/schedule | epsilon, squaredcos_cap_v2 DDIM, 원본 clipping·guidance 위치 |
| optimizer | AdamW, lr=1e-4, weight_decay=0 |
| LR/EMA | cosine, warmup 500, 원본 EMAModel power=0.75 |
| 기본 학습 예산 | 원본 20 epochs × 1,024 batches = 20,480 optimizer updates |
| effective batch | 원본 1,024, 작은 microbatch와 accumulation으로 맞춤 |
| native temporal window | 원본 resolved `rollout_steps` 확인·유지; 현재 기본값 8 |

주의: 현재 `config/train.yaml`의 기본 addon은 bimanual이다. quadruped addon을
명시적으로 선택해야 한다. 공통 DDIM 파일의 기본 5단계보다 hardware_gen의
100단계 override가 우선하므로, 파일 하나가 아니라 resolved config를 검사한다.
기존 `codesign/train_tt.py`의 간소화 모델이나 수집기의 H64/5-step `train` 명령은
이번 원본 TT 학습 경로로 사용하지 않는다.

긴 과제를 만들기 위해 window를 조용히 H64/500으로 늘리거나 이전 임의 stitching을
재사용하지 않는다. 원본의 window 실행·조건 갱신 경로를 확인하고, 10초 구간에서
E 고정·실제 초기 상태·절대 관절 목표·시간 연속성이 유지되는지 별도로 검증한다.
원본이 지원하지 않는 장기 생성 연결은 구현 차이로 명시하고 U/G/R에 동일 적용한다.

## 4. Reward-gradient 조건 G

원본 `t2.eval.diffusion_guidance`와 sampler를 사용한다. gradient는 원본과 같은
변수·좌표계·스케줄 계수에 적용하고, epsilon에 임의로 더하는 식으로 대체하지 않는다.
미분 가능한 task surrogate와 최종 실제 실행 reward는 정의·차이를 함께 저장한다.
MuJoCo 실행을 미분 가능하다고 가정하지 않는다.

개발 잡음 16개에서 scale={0, 0.1, 1, 10}을 비교한다. 후보 선택은 predicted reward가
아닌 실제 task return으로 한다. 동일 seed 3개에서 개발 평가하고 scale을 잠근 뒤
최종 평가에는 다시 맞추지 않는다. 이 단계는 별도 TT 학습이 아니라 생성·개발 평가다.

## 5. RL 잔차 조건 R

TT는 `eval`, 모든 θ의 gradient는 끈다. G의 reward-gradient는 R에서 끈다.
원본 ε 예측에 정책 잔차를 더하고 원본 sampler로 다음 latent를 계산한다.

    ε_used = ε_TT(z_k, condition, k) + mask_unknown × c × tanh(u_φ)
    z_(k-1) = original_scheduler_step(ε_used, k, z_k)

마스크는 원본 known/padding/inactive-hardware 제약을 그대로 따른다. c는 정규화된
prediction 공간에서 초기 0.1로 두고 개발 단계에서 확정한다. TT를 한 번만 호출해
캐시한 뒤 직접 x0를 복원하는 이전 방식은 쓰지 않는다.

정책은 원본의 **각 token hidden feature와 diffusion timestep**을 받아 modality별
작은 공유 head로 잔차를 출력한다. actor 입력을 구간 평균·표준편차로 축약하거나
시간축 전체에 같은 상수 보정을 강제하지 않는다. TT feature는 detach한다.
초기 평균 잔차는 0이다. PPO는 샘플링한 실제 확률변수의 log-prob와 applied residual을
함께 기록한다. masking/tanh 전후의 확률 정의를 연결 시험에서 검증한다.

PPO 연결은 기존 설치된 PPO 구현을 재사용한다. 별도 과제별 정책을 학습하며,
초기 설계 예산은 task당 seed 3개 × 1,000 updates × batch 16이다. 이는 제안 예산이며
실행 시간 약속이나 수렴 기준이 아니다. 먼저 5-update 연결 시험의 처리시간과
메모리를 측정해 전체 예상 비용을 보고하고 최종 학습 예산을 잠근다.
PPO 초깃값: lr=3e-4, clip=0.2, update epochs=4, gamma=1, GAE λ=0.95.

한 denoising 전이를 PPO의 한 전이로 기록한다. native window가 여러 개면 시간
window/denoising timestep을 함께 보존하고 전체 물리 에피소드를 하나의 rollout으로
연결한다. 중간 reward는 0, 실제 물리 실행 후 마지막 전이에 전체 task return을
지급한다. 낮은 terminal reward 자체와 구현 오류를 혼동하지 않는다.

연결 시험은 0 잔차 재현, 알려진 정답 잔차를 보상으로 주는 작은 문제의 학습,
PPO buffer의 terminal reward와 저장 궤적 재계산의 일치, 실제 모터 목표 변화를
포함한다. 이를 통과한 후 1,000회 학습을 진행한다.

## 6. 보상·평가·선택

과제는 straight, circle, zigzag, 실제 길이는 모두 10초/500 control steps다.
기존 과제 geometry와 진행·트랙 오차·직립·낙상·action-rate 항목을 읽어 물리 단위와
부호를 고정한다. 에피소드 도중 종료하더라도 부분 수행·종료 이유·생존 시간을
전부 남긴다. invalid S0는 성공/보상 0으로 세지 않는다.

PPO 보상은 **실제 MuJoCo task return**이다. 생성 S만 수정해서 얻는 predicted reward와
기존 `-5 × predicted/actual position MSE` 항목은 이번 PPO 보상에 넣지 않는다.
모델 오차는 별도 지표로 보고한다. RL이 task return을 높여도 dynamics 불일치가
자동으로 해결됐다고 주장하지 않는다.

개발 잡음과 별개인 고정 최종 잡음 64개를 과제·seed별 U/G/R 모두에 적용한다.
같은 noise seed는 같은 생성 출발점을 뜻하며, 보정 후 E까지 같다는 뜻은 아니다.
후보 수·초기화·physics·실행 길이·sampler calls를 맞춘다. 생성된 E가 학습 범위 안에
있는지와 초기 지지 상태를 기록한다. 평가 형상 20개에 E를 고정한 제어 일반화 시험은
자유 E 생성 시험과 구분해 별도 표로 보고한다.

정책/scale 선택은 개발 set만 사용한다. 최종 seed나 heldout 형상으로 재선택하지 않는다.
모든 후보를 평가하며 대표 영상은 전체 통계 뒤에 둔다.

필수 지표: actual task return, 완주율/생존시간, 트랙 내 시간, 진행거리·속도,
낙상·invalid 비율, predicted/actual 위치·관절 오차, 생성/학습 시간·메모리.
비교 예산은 sampler calls와 RL의 추가 학습 물리 에피소드를 각각 보고한다.
G와 R에 같은 학습 비용이 들었다고 표현하지 않는다.

## 7. 메모리·산출물·구현 순서

12GB GPU에서 먼저 microbatch 1로 확인하고 여유가 있으면 2/4로 늘린다. 실제
VRAM peak를 측정하며 목표는 9GiB 이하다. 원본 effective batch는 accumulation으로
유지한다. bf16 지원 여부를 점검하고, precision·accumulation 등 실행 변경을 기록한다.
데이터 loader는 workers=2로 시작하고 prefetch를 제한한다. 전체 작업은 CPU/RAM
상한이 있는 단일 서비스에서 순차 실행하며 GPU 학습 두 개를 동시에 켜지 않는다.
full-chain 시간은 사전학습 100 updates와 PPO 연결 시험을 측정한 후 산정한다.
전날의 수집 ETA를 TT/RL 학습 ETA로 재사용하지 않는다.

예정 출력 루트: `runs/tt_gradient_rl_chain_v1/`.

    data/       # source split, 토큰 schema·정규화·round-trip 검사
    tt_base/    # resolved config, raw/EMA, optimizer/scheduler/RNG 재개 상태
    gradient/   # U/G 개발 결과, scale 선택, 물리 궤적
    residual/   # 과제/seed별 PPO·critic·optimizer/RNG와 개발 결과
    evaluation/ # 잠근 U/G/R 최종 후보 전체 지표·궤적·영상·report.html
    chain.json  # 단계 상태·입출력 해시·종료 코드·실제 자원 사용

현재 재사용 가능한 것은 원본 t2 학습기·token adapters·guidance와 설치된 PPO,
실제 MuJoCo dynamics다. 필요한 새 구현은 **dynamics→원본 token 어댑터,
원본 sampler의 residual callback, 공통 물리 평가 연결, 순차 실행기**다.
기존 `train_diffusion_rl.py`는 축약 Net/42D state/구간 통계 정책에 결합돼 있으므로
명령만 체인으로 묶어 이번 설계가 구현됐다고 취급하지 않는다.

## 8. 구현과 실행

실행·재개·실측 결과는 [실행 기록](../runs/tt_gradient_rl_chain_v1/README.md)에 정리한다.
진입점은 `code/tt_dynamics/run_native_tt_chain.py`다.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/chain_mpl \
../transformer-transformer/.venv/bin/python -u \
  code/tt_dynamics/run_native_tt_chain.py run --device cuda
```

동일 명령을 메모리 제한 서비스에서 실행한다. 진행 중 별도 인스턴스를 켜지 않는다.
원본 native 시간 샘플러 확인 결과, 8시점은 연속 프레임이 아니라 에피소드 내 무작위
시점이다. 이번 생성은 원본 EvenTimestepSampler로 10초의 8시점을 고르고 관절 목표를
50Hz로 보간한다. 이는 U/G/R 공통의 명시적 실행 연결이며, 500시점 출력을 학습했다고
주장하지 않는다. 모델·adapter·loss는 원본을 사용하되, stock loader의 100-step 최소
길이 대신 8개의 서로 다른 시점이 가능한 유효 실패 자료를 보존한다. 실제 최단 길이는
11 step이라 40,000개 에피소드가 모두 포함된다.

마이크로배치는 VRAM 측정에 따라 1→2→4→8→16→32 중에서 결정한다. 원본 effective
batch 1,024와 업데이트 수는 유지한다. 데이터 loader는 재개 때 prefetch RNG를 잃지
않도록 main process에서 최대 두 개 mmap shard만 캐시한다. optimizer·scheduler·EMA·
RNG는 TT 10 update마다, PPO는 update마다 저장한다.
