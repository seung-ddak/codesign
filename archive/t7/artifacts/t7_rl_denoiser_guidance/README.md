# T7 Frozen TT Denoiser + RL Guidance

[Workflow](./rl_denoiser_guidance.html) — pretrained TT의 reverse process는 그대로 두고, 모든 100 denoising step에서 DSG의 gradient guidance를 RL residual policy `πφ`로 교체한다.

## 구현

- `stage2/code/train_rl_denoiser_guidance.py`: 동결 TT sampling, guided transition 기록, critic·actor 학습, history/checkpoint 저장
- `tt_patch/t2/eval/rl_guidance.py`: TT `guidance_fn`과 호환되는 `ResidualGuidancePolicy`, `RLResidualGuidance`, `GuidanceCritic`
- `stage2/code/design_scorer.py`: 최종 형상을 동결 `πθ`의 7-command 평균 locomotion return으로 채점

## 실제 흐름

1. Gaussian noise `x_T`에서 TT가 100-step reverse process를 수행한다.
2. 모든 100 step마다 `πφ(x_t,k)`가 stochastic residual action을 만들고 TT의 DSG guidance 자리에 들어간다.
3. 최종 hardware만 bank의 `(choice, seed)` evaluator adapter를 거쳐 MJX에서 평가된다.
4. terminal return으로 모든 guided state/action의 critic을 학습하고, 100 transition 평균 actor loss로 `πφ`를 한 번 갱신한다.
5. TT denoiser와 locomotion `πθ`는 optimizer에 포함되지 않는다.

## 검사와 실행

```bash
../transformer-transformer/.venv/bin/python stage2/code/train_rl_denoiser_guidance.py --self-check

# 학습은 사용자가 실행한다.
../transformer-transformer/.venv/bin/python stage2/code/train_rl_denoiser_guidance.py \
  --pi-theta stage2/runs/T5/try1 \
  --out stage2/runs/T7/rl_guidance_try1 \
  --updates 100 --batch 16 --denoise-steps 100 --guided-steps 100
```

현재 bank는 denoising search space가 아니라 고정 `πθ` evaluator가 이해하는 canonical robot으로 바꾸기 위한 최종 adapter다.
