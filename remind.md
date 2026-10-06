# Morphology–Behavior Co-design 연구 정본

> 2026-10-06 기준 현재 방향만 기록한다. 폐기된 결정과 과거 실험 원문은 `archive/remind_history.md`에 둔다.
> 신규 작업은 T번호나 stage 폴더를 만들지 않고 `code/`에서 진행한다.

## 1. 연구 질문

여러 4족 로봇 형상에서 수집한 범용 dynamics 경험을 학습한 Transformer-Transformer(TT)가, 학습 중 명시적으로 보지 않은 복합 task reward만으로 **형상과 action trajectory를 공동 생성**할 수 있는가?

- 최종 생성물: morphology `E*`, action trajectory `A*`, 예측 state trajectory `S*`
- 최종 검증: 같은 `(E*, s0, A*)`를 실제 MJX에서 실행해 예측 reward와 실제 reward 비교
- 범위: 시뮬레이션만 사용하며 실물 제작·Sim2Real은 제외
- 생성 형상 전용 PPO fine-tuning은 우선 진행하지 않음
- 생성된 것은 폐루프 정책 `π*`가 아니라 유한 길이 morphology-specific action trajectory임

## 2. 전체 파이프라인

| 순서 | 작업 | 핵심 산출물 | 통과 기준 |
|---:|---|---|---|
| 1 | 동작별 expert policy 구성 | locomotion·jump·stairs expert | 각 expert의 primitive gate 통과 |
| 2 | expert 전환 rollout 수집 | primitive 및 transition trajectory | 성공·실패·회복 전환이 모두 포함됨 |
| 3 | 여러 morphology·terrain의 rollout 통합 | morphology–state–action–next-state 데이터셋 | 형상·상태·행동·접촉 분포가 충분히 다양함 |
| 4 | TT joint dynamics 학습 | `p(E, S, A | s0, terrain)` 생성 모델 | held-out one-step·H-step 예측 gate 통과 |
| 5 | 보지 못한 복합 task reward guidance | guided `(E*, A*, S*)` | predicted reward가 unguided보다 높음 |
| 6 | 실제 MJX 재실행 | 실제 trajectory와 task reward | 실제 reward가 baseline보다 높고 예측-실제 gap 허용 범위 |

gate를 통과하지 못하면 다음 순서로 넘어가지 않는다. TT 예측이 실패하면 guidance보다 데이터 coverage와 horizon부터 수정한다.

## 3. Dynamics 수집용 expert policy

하나의 범용 정책을 계속 확장하지 않는다. 동작별 expert를 독립적으로 학습하고, 이를 전환하며 dynamics 데이터를 수집한다. expert는 최종 산출물이 아니라 다양한 물리 상태를 만드는 behavior policy다.

| Expert | 초기화·학습 범위 | 성공 기준 |
|---|---|---|
| Locomotion | T5를 257차원 관측으로 이식한 step-0 checkpoint를 동결 | 기존 7개 command의 생존·이동·회전 성능 유지 |
| Jump | T5에서 별도 분기하여 점프·착지 중심으로 학습 | 몸통 상승, 네 발 공중, 안정적 착지 후 생존 |
| Stairs | 별도 분기하고 낮은 계단부터 curriculum 적용 | 계단 진입·오르기·내리기 및 몸통 비접촉 생존 |

공통 원칙은 다음과 같다.

- expert 사이에 action space, robot state, morphology 표현을 동일하게 유지한다.
- Jump Expert는 정지 상태뿐 아니라 이동 중 점프와 착지를 포함한다.
- Stair Expert는 계단 위 동작뿐 아니라 평지에서의 진입과 평지 복귀를 포함한다.
- 정책이 전환되어도 MJX state를 reset하지 않고 연속으로 전달한다.
- 작은 action perturbation과 실패·충돌·회복 상태도 수집한다.
- 계단은 task ID만으로 처리하지 않고 terrain height observation을 사용한다.

### T5 가중치 이식 규칙

Jump·Stair Expert의 관측이 T5보다 확장될 때 기존 69차원 입력에 해당하는 가중치는 복사한다. 새 jump trigger·terrain height 입력의 첫 layer 가중치는 0으로 두고, observation normalization은 mean 0·variance 1로 초기화한다. 따라서 추가 학습 전에는 새 입력이 actor 출력에 영향을 주지 않고 T5와 같은 입력에 거의 같은 action을 출력해야 한다.

### Primitive transition 수집

expert scheduler가 아래와 같은 전환을 실행한다.

```text
locomotion -> jump -> landing -> locomotion
locomotion -> stairs -> locomotion
turn -> stairs
stairs down -> acceleration
```

`skill_id`와 command는 데이터 출처와 구간을 구분하는 metadata로 저장한다. TT가 실제 dynamics 대신 expert 이름에 의존하지 않도록 state·action·terrain·morphology를 핵심 물리 입력으로 사용한다.

## 4. 보지 못한 task의 의미

새 task는 학습 데이터의 물리적 support 안에 있지만, 그 목적 함수나 전체 조합은 학습하지 않은 task다.

| 데이터에 포함된 primitive·전환 | guidance에서 가능한 새 task 예시 |
|---|---|
| 전진·좌회전·우회전 | 지그재그·S자 경로 |
| 달리기·jump trigger·착지 | 특정 위치까지 달린 뒤 점프 |
| 평지↔계단·회전 | 지그재그 후 계단 오르기 |
| 계단 내려가기·평지 보행 | 계단을 내려간 직후 가속 |

데이터에 공중 상태나 계단 접촉이 전혀 없으면 reward gradient만으로 점프·계단 dynamics를 신뢰성 있게 만들 수 없다. 정확한 최종 task label은 필요 없지만 필요한 물리 primitive와 전환은 반드시 데이터에 있어야 한다.

## 5. rollout 데이터셋

각 sample/chunk에 다음 항목을 저장한다.

```text
morphology: choice 8 + continuous uniform 13
terrain: heightfield 식별자 또는 로컬 height observation
initial_state: s0
state: s_t
command / skill_id
action: a_t
next_state: s_{t+1}
contact
done / alive
```

- 128 morphology family를 모두 포함하고 family마다 여러 continuous seed를 사용한다.
- scheduler는 episode 중 expert를 전환하되 MJX state를 연속으로 유지한다.
- expert별 sample 수와 transition별 sample 수를 기록해 한 expert가 데이터셋을 지배하지 않게 한다.
- 성공 trajectory만 저장하지 않고 실패·충돌·회복도 포함한다.
- batch 결과는 즉시 저장하며 중단 후 이어서 수집할 수 있어야 한다.
- 데이터 수집 정책 reward는 primitive 습득용일 뿐 TT의 최종 task reward가 아니다.

## 6. TT 학습과 guidance

TT는 morphology·state·action trajectory의 joint distribution을 denoising으로 학습한다.

```text
condition/inpainting: terrain, initial state
denoise/generate: continuous morphology, state sequence, action sequence
family condition: discrete choice 8
```

- TT training loss는 denoising loss만 사용한다.
- 최종 task reward는 inference guidance에서만 사용한다.
- reward gradient는 morphology와 action trajectory에 적용한다.
- action limit, morphology 범위, state/contact consistency를 유지한다.
- predicted state를 실제 dynamics로 간주하지 않고 항상 MJX에서 동일 action을 재실행한다.

```text
TT(E*, s0, A*)  → predicted S*, predicted reward
MJX(E*, s0, A*) → actual S, actual reward
```

## 7. 평가 gate

### Expert policy

- Locomotion: 기존 T5의 7개 고정 command 기준선 유지
- Jump: 몸통 상승 + 네 발 공중 + 착지 후 생존을 모두 만족
- Stairs: 진입·높이 획득·완주 + 몸통 비접촉 생존
- Transition: policy 전환 전후 state가 끊기지 않고 착지·진입·복귀 구간이 실제로 포함됨

### TT dynamics

- held-out morphology의 one-step state 오차
- H-step state·contact·done 예측 오차
- predicted task reward와 실제 MJX reward의 순위상관
- action perturbation에 따른 next-state 변화 방향 일치

### Co-design

- guided > unguided TT
- guided > family 내 random search
- 같은 MJX 예산의 CMA-ES와 비교
- 여러 initial state에서 성능과 실패율 보고

## 8. 현재 근거와 폐기된 접근

### 유지할 기준선

- T5 try1: 128 choice, 2,457.6만 step, 평지 command 추종 학습 성공
- 고정 평가: 생존 84%, 이동 69%, 회전 56%
- 반복 평가 순위상관 0.913
- 원본 T5 checkpoint: `stage2/runs/T5/try1/checkpoints/000024576000`
- Locomotion Expert: `runs/expert/locomotion/checkpoints/000000000000`

### 단일 통합 정책 실패

- 기존 통합 정책은 T5와 같은 학습량으로 보행·점프·거친 지형을 처음부터 동시에 학습했다.
- 마지막 평균 episode 길이: T5 약 684 step, 통합 정책 약 215 step
- 대표 평지 전진 rollout: 약 10초 동안 0.19 m 이동(요구 약 4.95 m)
- T5 이식 후 짧은 jump 학습에서도 locomotion 성능이 하락했고 실제 점프는 학습되지 않았다.
- 측정 결과 jump reward가 기존 tracking reward를 압도한 것은 아니었다. 문제는 단일 정책에서의 최적화 간섭과 망각에 가깝다.
- 결론: T5는 Locomotion Expert로 동결하고 Jump·Stair Expert를 독립적으로 학습한다.

### 사용하지 않는 접근

- task-specific 정책 하나를 동결해 모든 형상을 채점하는 방식
- 생성 형상마다 PPO를 다시 fine-tuning하는 방식
- DDPO로 morphology를 직접 업데이트하는 방식
- UMI reference pose를 추종하는 reward
- 학습 데이터 support 밖의 동작을 reward guidance만으로 생성한다고 가정하는 방식

## 9. 코드 경계

- 신규 구현은 `codesign/code/`에 둔다.
- 원본 `transformer-transformer/`는 참고용으로 유지하고 직접 덮어쓰지 않는다.
- 기존 `stage2/`는 T5 checkpoint와 과거 실험 재현용으로만 유지한다.
- TT의 train·DiT·modality·denoising·guidance core는 우선 유지한다.
- 새로 필요한 구현은 expert별 학습 환경, expert scheduler가 포함된 rollout 수집기, dataset adapter, task reward guidance, MJX evaluator다.

## 10. 자원 운영

- GPU: RTX 4070 SUPER 12GB, RAM 30GB
- `MUJOCO_GL=egl`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MALLOC_ARENA_MAX=2`
- heavy run은 cgroup `MemoryHigh`·`MemoryMax`·`MemorySwapMax`를 설정한다.
- batch 결과는 즉시 저장하고 resume을 지원한다.
- available RAM < 8GiB 또는 swap > 512MiB이면 중단하고 batch/structure 단위를 줄인다.
- TT 학습은 batch 16~32에서 시작하고 필요하면 gradient accumulation을 사용한다.

## 11. 다음 작업

1. 완료: T5 checkpoint를 Locomotion Expert 기준선으로 확정
2. 진행 중: 매 episode jump trigger, trigger 중 tracking 보상 제거, 공중 후 착지 보상을 적용해 Jump Expert 독립 학습
3. Jump Expert gate 통과 후 Stair Expert의 낮은 계단 curriculum 설계
4. expert들이 각각 gate를 통과한 뒤 primitive transition scheduler 구현
5. dynamics rollout schema와 streaming 저장 형식 확정
