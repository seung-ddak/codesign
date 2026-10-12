# Jump Rollout TT · horizon 80

## 학습

- 데이터: `runs/dynamics/jump/batch_0000.npz`
- horizon: 80 step (1.6초)
- chunks: 735
- optimization: 1,000 step, batch 64
- loss: 1.5878 → 0.3685

## 착지 평가

### Reset부터 전체 생성 (`generated_jump_landing.json`)

45-step 준비 구간까지 action을 생성하면 예측 선택 후보는 MJX에서 유효한 점프를 만들지 못했다. 긴 open-loop action 생성이 초기 자세를 무너뜨렸다.

### Jump 시작 state 고정 (`generated_jump_landing_start45.json`)

실제 jump 시작 state와 jump command만 고정하고 이후 80-step action/state를 생성했다.

- 예측 선택 후보: candidate 5
- MJX 최대 상승: 0.254 m
- 이륙: step 8
- 최초 touchdown: step 13
- 총 네 발 체공: 66 step
- 안정 착지(네 발 연속 3 step 접지): 실패

8개 후보 모두 이륙 후 지면에 재접촉했다. 따라서 touchdown 생성은 통과했지만 균형 회복까지 포함한 stable landing은 통과하지 못했다. 원본 expert도 첫 touchdown 뒤 반복 체공하며 안정 착지 데이터가 부족하다.

## 재실행

```bash
MUJOCO_GL=egl ../transformer-transformer/.venv/bin/python code/evaluate_generated_jump.py \
  --data runs/dynamics/jump/batch_0000.npz \
  --checkpoint runs/tt/jump_seed0_h80/checkpoint.pt \
  --start 45 --candidates 8 \
  --out runs/tt/jump_seed0_h80/generated_jump_landing_start45.json
```

## Reward candidate guidance

32개 후보를 생성하고 TT 예측 reward가 가장 높은 candidate 28을 선택했다.

- unguided candidate 0 MJX reward: 4.547
- 후보 MJX reward 평균: 4.807
- guided candidate 28 MJX reward: 4.539
- MJX oracle candidate 25 reward: 8.831

예측 reward 선택은 무작위 후보와 후보 평균을 모두 개선하지 못해 실패했다. 실제로 안정 착지한 candidate 25가 있었지만 TT state/contact 예측 오차 때문에 선택하지 못했다.

`reward_guidance.npz`에는 reverse diffusion step `99, 90, ..., 0`의 action/state trajectory를 `denoising` 배열로 저장했다. 현재 morphology는 생성값이 아니라 `fixed_morphology`로 고정된 조건이다. Noise에서 morphology가 만들어지는 과정을 얻으려면 morphology를 denoising target에 포함한 joint TT 재학습이 필요하다.

## 영상

- `videos/candidate_28.mp4`: TT 예측 reward가 선택한 후보
- `videos/candidate_25.mp4`: 실제 MJX reward가 가장 높았던 안정 착지 후보
- `videos_guided/candidate_1.mp4`: 매-step gradient guidance 후 TT가 선택한 후보
- `videos_guided/candidate_6.mp4`: 동일 guided batch의 실제 안정 착지 후보

## 매-step gradient reward guidance

`evaluate_generated_jump.py --guidance-scale 0.2`는 100번의 reverse denoising마다 differentiable reward gradient를 action/future-state에 적용한다. 높이 reward는 0.6 m에서 부드럽게 포화시켜 비현실적인 높이로 reward를 속이는 현상을 줄였다.

동일한 noise seed와 후보 8개 비교:

| 지표 | Unguided | Guided 0.2 |
|---|---:|---:|
| 예측 선택 후보의 MJX reward | 4.613 | 4.901 |
| 후보 평균 MJX reward | 4.510 | 4.979 |
| 안정 착지 후보 | 없음 | candidate 6 |

매-step guidance 자체는 실제 MJX 결과를 개선했다. 다만 TT 예측 reward는 안정 착지 candidate 6 대신 candidate 1을 선택했으므로 ranking 문제는 남아 있다.

- 대조군: `gradient_guidance_baseline.json`
- guided 결과 및 denoising trace: `gradient_guidance_s02.json`, `gradient_guidance_s02.npz`
