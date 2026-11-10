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

## 최신 결과: 후보 8개 · 지그재그 100 iter (2026-11-05)

이전 4초 보고서는 `zigzag_1000iter` 폴더의 **후보 8개·100 iter** 완료 결과다. 기존 폴더명을 유지했으며 실제 설정은 `protocol.json`을 따른다. 학습 물리 에피소드 800개, 디노이징 전이 4,000개다. 평균 평가 보상 -20.634 → -17.345, 평균 전진 -0.164 → 0.142m, 생존 3/8 → 2/8이다. 보상과 전진은 개선됐지만 안정적인 트랙 추종 성공은 확인되지 않았다. 보고서는 전체 후보 8개 평가표와 대표 후보 0·1의 최신 내장 영상을 제공한다. 원형은 기존 2 iter 결과다.

직전 후보 2개·1,000 iter 실행은 300 iter에서 중지하고 `zigzag_1000iter_stopped_300updates.tar.gz`에 보관했다.

## 이전 결과: 후보 2개 · 지그재그 100 iter (2026-11-05)

`zigzag_100iter`는 기존 2 iter 검증과 동일한 후보 2개·200 상태·디노이징 5단계·seed 0으로 처음부터 100회 PPO 업데이트를 완료했다. 학습 물리 에피소드 200개, 디노이징 전이 1,000개다. 같은 평가 잡음에서 평균 실제 실행 보상은 -18.880에서 -25.658로 낮아졌고, 학습 후 두 후보 모두 낙상했다. 업데이트 수 증가만으로 성능 개선은 확인되지 않았다. 단일 seed·평가 2개에 한정된 결과다.

이 후보 2개 실험의 원자료는 `zigzag_100iter`에 보존한다. 원형 영상은 기존 2 iter 결과를 유지한다. 이전 지그재그 2 iter 원자료는 `validated_zigzag_seed0`에 보존되어 있다.

보고서 재생성: `python summarize.py`. 지그재그 영상 재생성: `python render_videos.py --tasks zigzag --zigzag-run zigzag_1000iter` (이 디렉터리 기준, 기존 Python 환경과 `MUJOCO_GL=egl` 사용).

## TT 예측 고정 + PPO 보정 학습 실험

`fixed_tt_zigzag_100iter`: `--fixed-tt-prediction --task zigzag --updates 100 --batch 8 --episode-states 501 --seed 0`.

초기 노이즈 후보 8개와 최고 노이즈 단계의 TT epsilon 예측을 고정한다. 각 PPO iter에서는 같은 예측에 새 확률적 보정값을 더하고 x0를 직접 복원하여 E·A를 CPU MuJoCo에서 최대 10초 (상태 501개·제어 500회·50Hz) 실행한다. E·S·A 보정 정책만 업데이트하며 TT를 다시 호출하거나 다음 노이즈 단계로 이동하지 않는다. 기존 보상·학습률·엔트로피는 유지한다. 이는 기존 5단계 DDIM과 다른 **한 단계 직접 복원 실험**이며 100단계 디노이징 실험이 아니다. TT checkpoint의 원래 5단계 스케줄 중 최고 노이즈 단계 계수를 사용한다.

TT 예측 묶음은 여러 window와 E prior용 호출로 구성되므로 raw forward 1회라는 뜻은 아니다. 학습 노이즈용 묶음 1회와 별도 평가 노이즈용 묶음 1회만 계산해 각각 재사용한다. `tt_prediction_audit.json`에 완료 시 기록한다. 학습 후보는 고정되지만 보정은 확률적으로 탐색한다. 보정값은 이전 iter에 누적하지 않고 같은 TT 예측에 매번 새로 적용한다. 평가는 학습과 다른 seed 1000000을 사용하며 보정 정책의 평균 행동을 사용한다.

`policy_best_training.pt`, `best_training.npz`, `best_training.json`은 최고 평균 **학습** 보상 시점의 정책·실제 궤적·후보별 수치다. 이 정책은 해당 rollout을 만든 PPO 업데이트 **직전** 정책이며, 최적 고정 평가 정책으로 간주하지 않는다. `policy.pt`는 마지막 업데이트 후 정책이다. 한정된 8개 노이즈에 과적합할 수 있으므로 학습 보상과 별도 평가를 구분해야 한다.

기존 4초 실행은 67 iter에서 중지해 `fixed_tt_zigzag_4s_stopped_67iter.tar.gz`에 검증 후 보관했다. 현재 같은 실험 폴더는 10초 설정으로 처음부터 학습하며 로그는 `fixed_tt_zigzag_10s_100iter.log`다. 낙상 시에는 10초 이전에 종료한다.

## 같은 몸체 E와 구간별 공유 S·A 보정 정책

두 10초 실험 모두 완료했다. 같은 별도 평가 후보 8개에서 기존 전체 구간 정책은 평균 보상 -20.770, 평균 전진 0.009m, 완주 0/8, 초기 상태 무효 1/8이다. 공유 구간 정책은 평균 보상 -21.339, 평균 전진 0.001m, 완주 0/8, 초기 상태 무효 0/8이다. 두 방식 모두 실행 직후 실패하는 결과로, 공유 구조의 성능 개선은 확인되지 않았다.

`shared_window_zigzag_10s_100iter`는 `--shared-window-actor`를 추가한 비교 실험이다. `fixed_tt_zigzag_100iter`의 10초 완료 실험과 동일하게 후보 8개, 정책 학습 100회, 상태 501개, 초기 잡음 번호 0, 별도 평가 잡음 번호 1000000, TT 예측 고정, 기존 보상·학습률·탐색 가중치를 유지한다.

- E 보정은 전체 구간의 평균 통계와 현재 E·잡음 단계에서 한 번 계산하며, 물리 모델은 전체 궤적에서 동일한 E를 사용한다.
- S·A 보정은 구간마다 **같은 정책 가중치**를 사용한다. 각 구간의 평균·표준편차, 전체 구간의 평균 통계, E·잡음 단계, 상대 구간 위치를 입력한다.
- 탐색 분산은 E 13개와 S·A 54개로 구성하고, S·A 분산은 모든 구간에 반복 적용한다. 분산을 공유해도 각 구간의 탐색 표본은 서로 다르다.
- 가치 함수는 전체 구간의 평균 통계를 사용한다. 정책과 가치 함수의 학습 파라미터 수는 구간 수와 무관하다.
- `DenoisingPolicy.load(..., episode_states=...)`로 다른 길이에서도 같은 공유 가중치를 불러올 수 있다. 기존 전체 구간 정책은 길이 불일치를 계속 거부한다. TT의 구간 길이와 디노이징 조건 검증은 유지한다.

현재는 구간들을 미리 보정해 전체 궤적을 만든 뒤 물리 실행한다. 실시간 상태 피드백이나 각 구간의 별도 보상, 다음 구간의 재생성은 추가하지 않았다. 길이를 바꿔 불러올 수 있다는 것은 긴 시간 성능을 보장한다는 뜻이 아니다. `generate_task.py`의 기존 다단계 디노이징 경로는 이번 한 단계 고정 예측 정책을 받지 않도록 단계 수 검증을 유지한다.

실행 위치: 작업 공간 루트.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl JAX_PLATFORMS=cpu \
  transformer-transformer/.venv/bin/python -u codesign/code/tt_dynamics/train_diffusion_rl.py \
  --task zigzag --updates 100 --batch 8 --episode-states 501 --seed 0 \
  --fixed-tt-prediction --shared-window-actor \
  --out codesign/runs/pipeline_v1/rl_denoising_v1/shared_window_zigzag_10s_100iter
```

비교 보고서 재생성: 이 폴더에서 `python compare_shared_window.py`. 보고서는 양쪽 실험의 전체 8개 평가 수치, 학습 곡선과 대표 영상을 사용한다. 영상은 초기 상태가 유효한 후보 중 최종 평가 보상이 가장 높은 후보를 표시한다. 이전 4초 보고서는 `report_4s_batch8.html`에 보관한다.

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


## 0단계 기록 확인

`code/tt_dynamics/audit_denoising_records.py`는 저장된 E와 생성 행동으로 물리를 다시 실행해 기존 보상·종료·상태 기록을 비교한다. 새 상세 기록은 저장된 qpos·qvel·실제 모터 목표·힘에서 단계별 보상도 재계산한다. 초기 무효, 낙상, 최대 시간 종료를 구분한다. 실제 학습 보상과 위치 일치 패널티, 초기 무효 패널티도 따로 저장한다.

전체 학습 궤적을 남기려면 학습 명령에 `--record-rollouts`를 추가한다. 후보 8개·10초·2회 검증은 `stage0_recording_check`에 보존했다. 이전 공유 정책과 첫 두 학습 보상이 정확히 일치해 기록 추가가 계산 결과를 바꾸지 않음을 확인했다. `ppo_terminal_rewards`는 실제 PPO 저장 버퍼의 마지막 단계 보상이며, 중간 보상이 0인지도 학습 중 검사한다. 정책 업데이트 전·후 번호와 원본 코드 해시·실행 환경 버전도 저장한다.

작업 공간 루트에서 검사 실행 예시:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl JAX_PLATFORMS=cpu \
  transformer-transformer/.venv/bin/python -u codesign/code/tt_dynamics/audit_denoising_records.py \
  --runs codesign/runs/pipeline_v1/rl_denoising_v1/zigzag_1000iter \
         codesign/runs/pipeline_v1/rl_denoising_v1/fixed_tt_zigzag_100iter \
         codesign/runs/pipeline_v1/rl_denoising_v1/shared_window_zigzag_10s_100iter \
         codesign/runs/pipeline_v1/rl_denoising_v1/stage0_recording_check \
  --out codesign/runs/pipeline_v1/rl_denoising_v1/stage0_record_audit_recheck
```

기존 검사 결과는 `stage0_record_audit/report.html`과 `audit.json`에 있다. 기존 실험 3개는 학습 로그 300회와 저장 궤적 재실행 96건에서 불일치가 없었지만, 매회 궤적과 상세 힘 기록이 없어 부분 확인이다. 새 2회 실행은 저장 자료만의 보상 재계산과 실제 재실행 비교까지 통과했다. 전체 재실행 144건에는 무보정·학습 전·최고 보상 등 중복 저장 기록이 포함된다. 독립적인 후보 144개를 의미하지 않는다.

보상·종료·모터 목표를 의도적으로 바꾼 기록이 검출되는 테스트를 포함해 관련 테스트 22개를 통과했다. 과제 성공이나 초기 물리 상태의 타당성은 다음 단계에서 검증해야 한다. 아직 전체 100회 상세 기록 학습을 새로 진행하지는 않았다.


## 1단계 물리 환경 일치 확인

`code/tt_dynamics/audit_denoising_physics.py`는 실제 `generate_task.py`의 `execute` 함수만 불러와 TT 생성이나 CLI 전체를 실행하지 않고, 같은 E·S0·절대 관절 목표를 CPU 학습 실행과 비교한다. 물리 실행 코드를 따로 복사해 만든 비교가 아니다. 초기 유효성 판정·초기 위치와 속도·실제 모터 목표·첫 제어·2초 구간·낙상 시점을 구분한다. 기본 목표 유지 4개, 이전 생성 행동 2개, 초기 무효 사례 1개, 최근 공유 정책 행동 1개를 사용했다.

작업 공간 루트에서 실행:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl \
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=.35 \
  transformer-transformer/.venv/bin/python -u codesign/code/tt_dynamics/audit_denoising_physics.py \
  --out codesign/runs/pipeline_v1/rl_denoising_v1/stage1_physics_audit_recheck
```

`stage1_physics_audit/report.html`에 비교 그래프·결과, `protocol.json`에 사전 허용 오차, 후보별 NPZ에 두 경로의 상태·관절 목표가 있다. 현재 CPU 학습·보고서 경로는 같은 함수를 사용하며 8개 모두 재현됐다. 별도 MJX 실제 궤적 7개 중 6개는 허용 오차를 통과했고, 1개는 2초 구간 자세 차이가 3.237도로 사전 기준 2도를 넘었다. 초기 무효 1개는 양쪽 모두 거부했지만 궤적 검증으로 세지 않는다.

차이가 있는 후보만 `--x64 --cases zigzag_1000iter_candidate0`로 별도 실행한 결과가 `stage1_physics_precision_check`다. 64비트에서도 자세 차이가 3.294도로 남아 정밀도만으로 설명되지 않았다. 이는 추가 진단이며 배포 기본 설정을 변경하지 않았다. 최근 공유 정책의 생성 행동은 두 경로 모두 0.04초에 넘어졌다. 같은 형상의 기본 관절 목표 유지와 비교한 초기 상태 차이는 약 0.00000018 이하이며, 기본 목표는 두 경로 모두 10초 유지됐다. 실제 행동과 물리 해석의 문제를 분리해야 한다.

현재 두 실행기의 초기 유효성 검사 규칙은 완전히 같지 않다. 이번 8개 입력에서 판정은 일치했지만 모든 입력의 동등성을 입증하지 않는다. 정지 유지 성공 또한 모션 추종이나 지그재그 수행 성공을 의미하지 않는다.
