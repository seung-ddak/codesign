# cp-design 연구

## 1. 연구 질문
- 4족 보행 로봇의 기본 동작 만을 이용하여 보지 못한 복합 task를 수행할 수 있는 최적의 형상과 정책을 만들 수 있을까?

## 2. 전체 파이프라인
    1. primitive motion datasets의 데이터를 랜더마이즈 하여 retargeting한다.
    2. retrageting하여 primivie motions tracking ppo expert
    3. primitive motions's transition도 ppo expert
    4. primitive motions motion tracking expert, transition expert로 여러 morphology에서 tracking rollout data 획득
    3. 여러 morphology rollout 통합 -> morphology–state–action–next-state dataset 생성
    4. TT joint dynamics 학습 -> p(E, S, A | s0, terrain) 생성 모델, held-out one-step·H-step 예측
    5. 보지 못한 복합 task reward guidance


## 3. Dynamics 수집용 expert


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
