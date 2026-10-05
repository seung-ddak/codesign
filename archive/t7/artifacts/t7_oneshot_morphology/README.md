# T7 One-shot Morphology RL

[구현 해설](./t7_oneshot_morphology.html) — noise latent에서 topology와 continuous morphology parameter를 한 번에 sampling하고, 동결된 locomotion `πθ`의 실제 MJX return으로 one-shot policy `πφ`를 갱신하는 과정을 설명한다.

## 구성

- `stage2/code/train_oneshot_morphology.py`: topology categorical + 13개 Beta distribution, horizon-1 PPO, 기록과 checkpoint 저장
- `stage2/code/locomotion_env.py`: 기존 `(choices, seed)`와 직접 `(choices, uniforms)` 입력을 모두 지원
- `stage2/code/design_scorer.py`: direct uniforms로 만든 바로 그 형상을 동결 `πθ`와 7개 고정 명령으로 평가
- `one_shot_morphology.html`: 전체 workflow 다이어그램

TT decoder, 100-step denoising, RoboToken nearest-bank projection은 사용하지 않는다. Canonical generator가 열거한 `8-bit + 13-uniform` 형상 128개 topology만 허용하므로 형상은 항상 기존 4족 보행 범위 안에서 생성된다.

## 검사와 실행

```bash
# 구조 검사만 수행하며 학습하지 않는다.
../transformer-transformer/.venv/bin/python stage2/code/train_oneshot_morphology.py --self-check

# 1차 smoke — 실제 실행은 사용자가 수행한다.
../transformer-transformer/.venv/bin/python stage2/code/train_oneshot_morphology.py \
  --pi-theta stage2/runs/T5/try1 \
  --out stage2/runs/T7/oneshot_try1 \
  --updates 20 --batch 16 --ppo-epochs 1
```

현재 상태는 구조 구현과 self-check 완료이며, 학습 결과는 아직 없다.
