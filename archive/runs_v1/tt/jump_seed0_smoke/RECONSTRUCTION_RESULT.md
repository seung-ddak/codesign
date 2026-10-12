# Dynamics Reconstruction 평가 결과

## 평가 방법

학습된 TT에 다음 정보를 고정해 32-step trajectory의 미래 31 step을 복원했다.

- morphology
- 최초 `qpos`, `qvel`, `contacts`
- 전체 expert `action`, `command` 시퀀스

TT 예측을 미래 state가 최초 state에서 변하지 않는 persistence baseline과 비교했다. 평가 샘플은 64개다.

## 결과

| 지표 | TT | Persistence | TT 개선율 |
|---|---:|---:|---:|
| qpos MAE ↓ | 0.3404 | 0.4348 | 21.72% |
| qvel MAE ↓ | 4.6181 | 5.7527 | 19.71% |
| contact accuracy ↑ | 83.06% | 76.40% | +6.67%p |

TT가 세 지표 모두 persistence baseline보다 좋다. 따라서 morphology, initial state와 expert action/command를 조건으로 단순히 초기 state를 유지하는 것보다 rollout의 미래 변화를 더 잘 복원했다.

## 산출물

- 원본 평가 수치: `reconstruction.json`
- 평가 코드: `code/evaluate_rollout_tt.py`

## 해석 범위

평가 샘플은 학습에 사용된 동일한 `batch_0000.npz`에서 추출됐다. 따라서 이 결과는 학습 데이터 reconstruction gate 통과를 뜻하며 일반화 성능의 증거는 아니다. 다음 평가는 별도 seed로 수집한 rollout batch를 고정 validation/test set으로 사용해야 한다.
