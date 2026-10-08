# Jump Rollout TT — seed 0 smoke

## 결과 아티팩트

- [통합 HTML 리포트](report.html)
- [학습 결과](TRAINING_RESULT.md)
- [Dynamics reconstruction 평가 결과](RECONSTRUCTION_RESULT.md)

## 목적

Jump expert rollout NPZ가 morphology-conditioned trajectory diffusion 학습에 연결되는지 확인하는 overfit smoke test다. 원본 T2 hardware-token 모델이나 reward guidance 결과가 아니다.

## 데이터

- 입력: `runs/dynamics/jump/batch_0000.npz`
- 128 morphology family, family마다 continuous seed 1개
- rollout 길이 500 step, control 주기 0.02 s
- 32-step non-overlapping chunk 1,846개
- morphology condition: choice 8 + uniforms 13 = 21차원
- trajectory: qpos 19 + qvel 18 + action 12 + command 5 + contacts 4 = 58차원

## 모델과 결과

- 3-layer Transformer encoder, width 128, 4 heads
- DDPM noise step 100
- batch 64, AdamW, learning rate 3e-4
- 학습 1,000 step
- denoising loss: 1.6654 → 0.4090
- GPU: RTX 4070 SUPER, 약 1.7 GB

학습 loss 감소로 NPZ adapter와 denoising loop가 작동함은 확인했다. 미래 dynamics 예측, unseen morphology 일반화, MJX 일치, action/morphology 생성은 아직 검증하지 않았다.

## Reconstruction 평가

실제 morphology, initial state, expert action/command를 고정하고 미래 qpos/qvel/contact를 inpainting한다.

```bash
../transformer-transformer/.venv/bin/python code/evaluate_rollout_tt.py \
  --data runs/dynamics/jump/batch_0000.npz \
  --checkpoint runs/tt/jump_seed0_smoke/checkpoint.pt \
  --out runs/tt/jump_seed0_smoke/reconstruction.json
```

현재 batch는 학습에도 사용됐으므로 이 평가는 reconstruction gate일 뿐 일반화 평가가 아니다. 다음 continuous seed batch를 validation/test로 수집해야 한다.

## MJX 재실행 평가

저장된 state와 expert action으로 MJX one-step dynamics를 재실행한다. 결과는 `mjx_replay.json`에 저장된다.

```bash
MUJOCO_GL=egl ../transformer-transformer/.venv/bin/python code/validate_mjx_replay.py \
  --data runs/dynamics/jump/batch_0000.npz \
  --samples 0,32,64,96 \
  --out runs/tt/jump_seed0_smoke/mjx_replay.json
```

sample 0 CPU 검증은 qpos MAE 2.75e-5, qvel MAE 0.00284, contact accuracy 99.95%로 통과했다. 연속 open-loop는 GPU batched 수집과 CPU single-env 재실행 사이의 작은 접촉 수치 차이가 누적되어 별도 진단값으로 기록한다.

## 조건부 jump action 생성 평가

Morphology, jump 시작 state, jump command를 고정하고 action/state 후보 8개를 생성한 뒤 MJX에서 실행한다.

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl ../transformer-transformer/.venv/bin/python \
  code/evaluate_generated_jump.py \
  --data runs/dynamics/jump/batch_0000.npz \
  --checkpoint runs/tt/jump_seed0_smoke/checkpoint.pt \
  --candidates 8 \
  --out runs/tt/jump_seed0_smoke/generated_jump.json
```

- 예측만으로 선택한 candidate 5: MJX 상승 0.194 m, 네 발 체공 5 step
- MJX 결과상 최상 candidate 7: 상승 0.355 m, 네 발 체공 17 step
- 결과: `generated_jump.json`, 전체 trajectory: `generated_jump.npz`

현재 32-step(0.64초) horizon으로 이륙과 체공이 나왔으므로 smoke 단계에서는 horizon을 늘리지 않았다. 착지까지 생성·평가하려면 80~128 step checkpoint가 필요하다. 예측 최상 후보와 MJX 최상 후보가 달라 TT state prediction을 후보 ranking에 바로 쓰기에는 아직 오차가 크다.
