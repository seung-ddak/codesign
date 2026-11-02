# 실제 초기 상태 × 행동 출처: 2×2 비교

[판정] 주 실험은 공통 생존 표본 부족으로 형상 예측 학습 비교 불가. 주 기준은 바꾸지 않았고 회귀 학습과 TT 재학습은 실행하지 않았다.

## 사전 고정한 실험

실제 MJX GPU, 16형상, 19모션×4구간=76개 행동 쌍, 1초(50 control step), kp200. 실제 출발 상태는 형상별 리타겟 레퍼런스와 home 목표각 1초 시뮬레이션 두 가지다. 행동은 e000의 레퍼런스 목표각과 기존 PPO 목표각 두 가지다. 절대 목표각 열은 전 형상·출발 조건에 동일하게 적용한다. 동일 형상·그룹에서 행동 출처만 바꿀 때 qpos/qvel은 동일하다.

PPO source는 기존 kp30 수집 데이터다. 모션별로 유효하고 모션 전환이 없는 연속 50step 구간을 고른다. 구간 수가 가장 많은 source episode에서 고르게 네 구간을 택한다. source episode·시작 프레임·실제 목표각은 action_bank.npz와 conditions.json에 동결되어 있다.

주 평가 표본은 네 조건과 모든 16형상에서 1초 동안 생존한 동일 그룹의 교집합이다. 최소24그룹·8모션이 필요하다. MLP의 안 본 형상 vx 오차가 그룹 평균 B0보다20% 이상 줄고, 효과 부호80% 이상 일치하며 효과 표본이32개 이상이어야 통과한다. s0는 항상 실제 초기 상태다. 형상 좌표 E를 쓰는 모델과 E 없이 s0+A를 쓰는 모델을 함께 비교한다. 조건별 모든 형상 생존 그룹은 보조 평가이며 주 표본을 대체하지 않는다.

## 결과

총4,864회 수집 완료. 조건별 생존은 아래와 같다.

| 출발 / 행동 | 생존 | 생존율 |
|---|---:|---:|
| 레퍼런스 / 레퍼런스 | 1017/1216 | 83.6% |
| 레퍼런스 / PPO | 483/1216 | 39.7% |
| home1초 / 레퍼런스 | 929/1216 | 76.4% |
| home1초 / PPO | 388/1216 | 31.9% |

네 조건·16형상 공통 생존 그룹은0개. 개별 조건에서도 모든 형상에서 살아남은 그룹은3/0/0/0개로 부족했다. 데이터 수집의 완료와 회귀 평가의 가능 여부는 구분한다. results.json의 회귀 지표가 없는 것은 실행 누락이 아니라 사전 표본 기준 미달이다.

## 사후 재생 검증 (주 실험과 분리)

공통 표본 부족을 확인한 뒤 e000에서만 같은76개 PPO 행동을 실제 source 상태·레퍼런스 상태·home1초 상태에서 kp30/200으로 재생했다. 별도 posthoc_probe_conditions.json에 실행 전 조건을 기록했다.

원본 실제 상태·kp30에서는76/76 생존했고, 원본과 전체 관절각 RMSE0.00001842rad, 마지막 XY 오차 중앙값0.00001060m로 궤적이 복원됐다. 같은 상태와 행동에서 kp만200으로 바꾸면37/76 생존했다. 레퍼런스/home 출발도 e000의 kp30에서는 모두76/76 생존했다.

이 결과는 목표각이 생성된 제어 이득과 재생 이득의 불일치를 보여준다. PPO 행동이 전 형상에서 우수/열등하다거나 초기 자세가 형상 학습을 결정한다는 결론은 내리지 않는다. 다른15형상의 kp30 비교는 미실행이다. qpos/qvel만 복원했고 원본 solver warm-start 상태는 저장되어 있지 않다는 한계가 있다.

## 재현

저장소 루트에서 실행한다. GPU가 보이는 일반 터미널과 `../transformer-transformer/.venv` 환경이 필요하다. 기본 실험은 watchdog으로 감시하며 JAX GPU pool40%, CPU2스레드, 임시 파일·캐시는 /home 아래로 제한한다. 파일과 조건이 달라지면 hash 검증으로 중단한다. 기존 수집 파일은 fingerprint 검증 후 재사용한다. 변경 실험은 새 폴더를 사용한다.

```bash
TMPDIR="$PWD/.runtime/tmp" ../transformer-transformer/.venv/bin/python -u runs/pipeline_v1/factorial_init_action/run.py

env -u JAX_PLATFORMS CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.4 \
  TMPDIR="$PWD/.runtime/tmp" MPLCONFIGDIR="$PWD/.runtime/mpl" XDG_CACHE_HOME="$PWD/.runtime/cache" \
  ../transformer-transformer/.venv/bin/python -u runs/pipeline_v1/factorial_init_action/run.py --worker posthoc_probe

MPLCONFIGDIR="$PWD/.runtime/mpl" ../transformer-transformer/.venv/bin/python runs/pipeline_v1/factorial_init_action/finalize.py
```

초기 수집 약110초. 데이터와 결과 약45MiB. 주 실험에서 전체 GPU 사용 관측 최대1647MiB, RAM 여유 최소15.6GiB. 사후 소규모 검증은 별도 실행으로, 주 watchdog 자원 로그에 포함하지 않는다. 관련 코드 검증17개와 전체 데이터의 형상·유한값·동일 행동·동일 초기 상태·해시 검증을 통과했다.

report.html은 최종 보고서다. conditions.json/입력 해시, action_bank.npz, rollouts/*/factorial.npz, collection_audit.json, results.json, summary.json, 사후 검증 조건·결과·궤적을 함께 보관한다. predictions.npz는 회귀가 실행되지 않아 비어 있다. 이전 단계의 comparison.png는 회귀값이 없는 초기 그림이며, 최종 보고서는 final_comparison.png를 사용한다.
