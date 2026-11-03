# PPO replaces reward-gradient guidance inside joint diffusion

이번 구현은 diffusion 내부의 guidance를 PPO로 바꾼다. 이전 `task_rl_v1/v2`의 별도 폐쇄 루프 관절 제어기와 다르다.

1. 동결된 TT가 noisy E와 각 window의 noisy S/A를 denoise한다.
2. PPO는 window별 noisy S/A의 채널 평균·표준편차, noisy E, diffusion 시간을 관측한다.
3. PPO가 **E 좌표별·window의 S/A 채널별 epsilon 보정**을 출력한다. `0.25*tanh(action)`로 제한하며 window 안에서는 시간축에 같은 보정을 적용한다.
4. 기존 DDIM 갱신이 그 보정으로 E와 S/A latent를 모두 갱신한다. known S0·window 경계 마스크, E 범위 및 비활성 좌표 고정은 유지한다.
5. 마지막 생성 E와 A를 실제 CPU MuJoCo에서 open-loop 실행한다. 매 물리 step에서 별도 PPO 제어기를 호출하지 않는다.
6. `dense_task_reward`의 실제 실행 return에서 **생성 S와 실제 S의 위치 MSE ×5**를 빼 terminal reward로 사용한다. invalid start는 -20, 실제 낙상 -20은 dense return에 포함된다. 생성 상태만 좋게 꾸며 보상을 얻는 경로를 줄이기 위해 상태 오차도 감점한다.
7. 한 diffusion 단계가 PPO transition 하나다. 중간 보상은 0, 마지막 단계에 실제 물리 보상을 주며 gamma=1이다. 학습 잡음과 평가 잡음을 분리한다.

E는 고정하지 않는다. 미래 S·A도 공동 생성한다. 초기 S0만 기존 v16과 동일하게 E의 home 자세 1초 안착 결과로 정한다. S0 자체를 자유 생성하는 구현은 아니다. 모델 기반 생성 상태와 실제 실행 상태는 별도로 저장한다.

RL 경로는 `guidance_scale=0`이고 `torch.autograd.grad(reward,...)`를 호출하지 않는다. TT 가중치는 고정하며 PPO policy만 학습한다. 기존 gradient guidance는 옵션을 지정하지 않았을 때 그대로 작동한다. PPO epsilon 보정과 reward gradient를 동시에 켜면 오류로 중단한다.

## 코드

- `code/tt_dynamics/joint_diffusion_guidance.py`: denoising callback과 RL audit. 기존 sampler 재사용.
- `code/tt_dynamics/rl_diffusion_guidance.py`: residual PPO policy, transition 수집, 저장/로드 및 shape 검사.
- `code/tt_dynamics/train_diffusion_rl.py`: 실제 TT 공동 생성 → MuJoCo 보상 → PPO 학습.
- `code/tt_dynamics/generate_task.py --rl-guidance-policy`: 기존 생성·MJX 평가 경로에서 학습된 policy 사용.

## 실행

저장소 루트에서 CPU 학습:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl transformer-transformer/.venv/bin/python -u codesign/code/tt_dynamics/train_diffusion_rl.py \
  --task zigzag --updates 100 --batch 8 --episode-states 200 \
  --out codesign/runs/pipeline_v1/rl_denoising_v1/train_zigzag_seed0
```

기존 생성기로 추론·MJX 실행 (`codesign` 디렉터리에서, GPU가 없다면 `--device cpu`):

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl MPLCONFIGDIR=/tmp/rl_denoising_mpl ../transformer-transformer/.venv/bin/python code/tt_dynamics/generate_task.py \
  --ckpt runs/pipeline_v1/fast_mode24_v16_twin/tt_joint \
  --rl-guidance-policy runs/pipeline_v1/rl_denoising_v1/validated_zigzag_seed0/policy.pt \
  --expert-config runs/pipeline_v1/experts_wide16/walkForwardSlow__r00 --kp 30 \
  --task zigzag --seconds 4 --candidates 2 --seed 1000000 --scales 0 \
  --rest-s0 --joint-full-guidance --e-rest-s0 --no-s0-align --no-render \
  --device cpu --out runs/pipeline_v1/rl_denoising_v1/generator_cpu_check
```

Policy의 task·TT checkpoint hash·episode 길이·horizon·diffusion step 수·기본 task geometry를 검증한다. 모델/길이를 바꿀 때 새 policy를 학습해야 한다. CPU 학습과 MJX 추론은 backend 차이가 있으므로 결과가 동일하다고 가정하지 않는다.

## 검증 범위

`validated_zigzag_seed0`: 실제 v16 twin checkpoint, 200 상태(약4초), 2후보, PPO 2update. 4개 학습 물리 episode·20개 denoising transition의 **구현 실행 검증**이다. 동일한 별도 평가 잡음에서 unguided, reward-gradient(scale10), 학습 전/후 RL을 비교한다. 이 작은 실행으로 task 성능이나 수렴을 판정하지 않는다.

한계: window당 채널별 상수 epsilon 보정으로 행동 공간을 줄였다. 더 세밀한 token별 조절은 구현하지 않았다. 초기 S0의 연속 미분은 필요하지 않으며, policy gradient는 실제 terminal return으로 계산한다. E·S·A 보정이 비영이라고 해서 실제 task 성능이 좋아졌다는 뜻은 아니다.

기존 생성기 통합 검증도 `generator_cpu_check/summary.json`에 완료했다. CPU MJX에서 두 후보의 실제 실행까지 수행했으며 RL audit에 reward gradient는 없고 E/S/A epsilon 보정은 모두 기록됐다. CPU MuJoCo 학습 재생과 MJX 실행 수치는 backend 차이로 조금 다르다. GPU 실행에서는 위 CPU 전용 `JAX_PLATFORMS=cpu`와 `--device cpu`를 제외한다.

## RL-only track videos

보고서 영상은 학습 후 diffusion PPO만 표시한다. 지그재그와 원형 각각 평가 후보 2개를 모두 포함하며, `videos/<task>_candidate<index>_rl_track.mp4`를 절대 경로로 연결한다. 노란 중심선·흰 경계는 각 task의 `guide_lines()`에서 가져와 시작 좌표계에 맞춰 렌더링한 표시다. 물리 충돌 객체는 추가하지 않았다.

원형은 `validated_circle_seed0`에서 지그재그와 같은 2update·2후보·200 상태 조건으로 별도 PPO 학습 및 실제 실행을 수행했다. 후보별 생성 E와 저장된 `rl_after.npz`의 실제 qpos를 사용하며, 기존 gradient 비교 영상은 보존하되 보고서에는 표시하지 않는다.
