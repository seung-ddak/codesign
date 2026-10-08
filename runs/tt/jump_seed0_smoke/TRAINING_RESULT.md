# Train Rollout TT 결과

## 실행 구성

| 항목 | 값 |
|---|---:|
| 입력 rollout | `runs/dynamics/jump/batch_0000.npz` |
| morphology 수 | 128 |
| 학습 chunk 수 | 1,846 |
| chunk horizon | 32 steps |
| trajectory 차원 | 58 |
| morphology condition 차원 | 21 |
| 학습 step | 1,000 |
| batch size | 64 |
| seed | 0 |
| device | CUDA |

모델은 morphology를 조건으로 `qpos(19) + qvel(18) + action(12) + command(5) + contacts(4)` trajectory에 추가된 diffusion noise를 복원하도록 학습했다.

## 학습 결과

| 지표 | 값 |
|---|---:|
| 최초 loss (step 1) | 1.6654 |
| 최저 기록 loss (step 700) | 0.3350 |
| 최종 loss (step 1,000) | 0.4090 |
| 마지막 5개 기록 평균 loss | 0.4087 |
| 최초 대비 최종 loss 감소 | 75.44% |

학습 loss가 안정적으로 감소했으므로 rollout NPZ 로딩, morphology conditioning, trajectory denoising 학습 및 checkpoint 저장까지 연결된 것을 확인했다. 최저 loss 이후 변동은 있으나 마지막 구간은 약 0.4 수준이다.

## 산출물

- 설정: `config.json`
- 원본 학습 기록: `progress.jsonl`
- 모델: `checkpoint.pt`

## 해석 범위

이 결과는 하나의 jump rollout batch에 대한 denoising smoke test다. 아직 unseen morphology/seed 일반화, MuJoCo 재실행 일치, reward-guided morphology/action 생성 성능을 의미하지 않는다.
