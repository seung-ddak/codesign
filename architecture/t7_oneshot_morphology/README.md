# T7 One-shot Morphology RL

[Workflow](./one_shot_morphology.html) — latent noise에서 valid topology와 13개 uniforms를 한 번에 sampling하고, 동결 `πθ`의 locomotion return으로 `πφ`만 갱신한다.

## 구현

- `stage2/code/train_oneshot_morphology.py`: one-shot policy, horizon-1 PPO, history/checkpoint 저장
- `stage2/code/locomotion_env.py`: 기존 seed 입력과 호환되는 direct uniforms 입력
- `stage2/code/design_scorer.py`: `(choices, seed)`와 `(choices, uniforms)`를 동일 frozen-policy evaluator로 채점

## 검사와 실행

```bash
../transformer-transformer/.venv/bin/python stage2/code/train_oneshot_morphology.py --self-check

# 학습은 사용자가 실행한다.
../transformer-transformer/.venv/bin/python stage2/code/train_oneshot_morphology.py \
  --pi-theta stage2/runs/T5/try1 \
  --out stage2/runs/T7/oneshot_try1 \
  --updates 20 --batch 16 --ppo-epochs 1
```

기본 구조는 TT decoder, DDPO denoising trajectory, RoboToken nearest-bank projection을 사용하지 않는다.
