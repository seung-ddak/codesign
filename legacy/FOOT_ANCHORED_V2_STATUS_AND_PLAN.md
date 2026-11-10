# Foot-Anchored v2: 진행 기록과 V100 전환 계획

## 현재까지 확인한 결론

wide16 action 실험은 16개 형상(`e000`, `e001`, `e002`, `e004`~`e016`)에서 완료했다. Orig-only 모델과 Orig·Swap·Offset 모델 모두 20,000 optimizer step을 완료했고, D5와 D4 결과를 저장했다.

Action Intervention은 Offset 명령에 대한 예측을 크게 개선했다. 관절각 ΔMAE는 평균 약 44% 줄었고 slope 중앙값은 약 0.49에서 0.87로 증가했다. 반면 Swap은 Yaw·높이·관절각에서 일관된 개선을 보이지 않았다.

D4에서는 X 위치, 높이, 관절각 ΔMAE가 일부 감소했지만 Yaw slope가 거의 0에 머물렀다. 따라서 형상 E의 물리적 효과를 TT가 충분히 학습했다고 판단하지 않는다. 기존 D4는 형상별 높이 정렬과 초기 접촉 차이가 섞였고, 세 형상 모두의 접촉이 일치한 그룹도 매우 적었다.

결론은 형상 수를 늘리는 것만으로는 형상 민감도 문제가 해결되지 않는다는 것이다. 동일한 S₀·A에서 E만 변경한 실제 MJX rollout을 학습 데이터에 추가해야 한다.

## Foot-Anchored v2 구현

`foot_anchored_s0.py`는 기본적으로 32개 연속 형상과 형상당 8개 초기 상태를 생성한다. 세 개의 arm-free 불활성 E 좌표는 고정하고, 네 발 접촉·관절 한계·자체 충돌·지면 관통·지지 다각형·짧은 정지 검사를 수행한다.

`collect_foot_anchored.py`는 각 형상에서 Orig·Swap·Offset을 실행하고, clipping 이후 Action RMS가 작은 그룹을 제외한다. 동일 qpos·qvel·A를 가까운 다른 형상에 적용하는 Morphology Intervention도 수집한다. 초기 상태는 CPU MuJoCo와 MJX에서 모두 검사한다.

메모리 문제를 줄이기 위해 rollout을 작은 batch로 나누고, 형상 환경을 하나씩 생성한다. 형상 처리 후 JAX cache와 Python 객체를 정리한다. 기본 batch는 3이며 `FOOT_ROLLOUT_BATCH=1`로 더 낮출 수 있다.

`run_foot_anchored_pipeline.sh`는 `prepare → dynamics → joint → guidance` 순서로 실행된다. 수집 완료 전에는 `prepare.done`을 만들지 않으며, Dynamics gate가 없으면 Joint 학습을 시작하지 않는다. Joint gate는 서로 다른 후보의 개별 성공률을 합치지 않고 같은 후보가 초기 물리 유효성·생존·경로 일치를 모두 만족하는지 계산한다.

## 현재 중단 지점

`runs/pipeline_v1/foot_anchored_v2/s0.npz`는 32개 형상 × 8개 상태로 생성되었다. 그러나 기존 컴퓨터에서는 `collect_foot_anchored.py`가 운영체제의 `Killed` 종료를 받아 `data/`와 `prepare.done`을 만들지 못했다. 이는 학습 실패가 아니라 MJX/JAX rollout 수집 중 메모리 문제다.

## V100에서 재개할 순서

V100에서는 먼저 CUDA·JAX 환경과 VRAM을 확인한 뒤 같은 출력 경로에서 수집을 재개한다. 기존 `s0.npz`는 재사용하고, 비어 있는 `data/`만 채운다. Foot-Anchored 스크립트의 기본 expert 설정은 저장소에 있는 `runs/pipeline_v1/experts_wide16/walkForwardSlow__r00`을 사용한다. 다른 설정을 쓸 때만 `FOOT_EXPERT_CONFIG`로 경로를 지정한다.

```bash
cd /home/cgr-ugrad-2026/quad_ws/codesign
FOOT_ROOT=runs/pipeline_v1/foot_anchored_v2 \
FOOT_EXPERT_CONFIG=runs/pipeline_v1/experts_wide16/walkForwardSlow__r00 \
FOOT_ROLLOUT_BATCH=3 \
bash code/tt_dynamics/run_foot_anchored_pipeline.sh prepare
```

메모리 부족이 계속되면 batch를 1로 낮춘다.

```bash
FOOT_ROOT=runs/pipeline_v1/foot_anchored_v2 \
FOOT_ROLLOUT_BATCH=1 \
bash code/tt_dynamics/run_foot_anchored_pipeline.sh prepare
```

`prepare.done`과 `data/morph_*.npz`가 생성된 뒤에만 다음 단계를 실행한다.

```bash
FOOT_ROOT=runs/pipeline_v1/foot_anchored_v2 bash code/tt_dynamics/run_foot_anchored_pipeline.sh dynamics
FOOT_ROOT=runs/pipeline_v1/foot_anchored_v2 bash code/tt_dynamics/run_foot_anchored_pipeline.sh joint
FOOT_ROOT=runs/pipeline_v1/foot_anchored_v2 bash code/tt_dynamics/run_foot_anchored_pipeline.sh guidance
```

## 실험 판정 기준

1. 모델 A: Orig-only.
2. 모델 B: Orig·Swap·Offset.
3. 모델 C: 모델 B와 Morphology Intervention.
4. D5에서 Offset과 Swap의 행동 변화 slope·ΔMAE를 모델 A와 비교한다.
5. D4는 가능한 pairwise contact-matched 공통 S₀에서 E만 바꿔 평가한다.
6. D4와 D5 결과가 Validation 기준을 통과한 경우에만 Joint Generation을 진행한다.
7. 32-state 공동 생성이 통과한 뒤에만 10초 Reward Guidance를 평가한다.

## 아직 구현하지 않은 범위

- 19개 motion expert 전체를 action bank에 합치는 작업
- 10초 전체 reward를 하나의 E에 역전파하는 shared-E 최적화
- 생성 S₀에 대한 모든 관절 한계·자체 충돌 검사의 완전한 통합
- 실제 V100에서의 benchmark와 전체 32개 형상 수집

이 항목들은 Dynamics 수집과 D4·D5 검증이 성공한 뒤 진행한다.
