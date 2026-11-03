# Task RL v2: dense physical reward

기존 실험의 형상 E·초기 상태·12/4 학습/평가 분할·PPO 구조·seed0·4초 horizon·관절 제어를 유지하고, 학습 보상을 변경했다. 기존 prefix 보상은 기본 옵션으로 유지한다. 새 실험은 `--reward-version dense`를 지정한다.

매 0.02초 실제 물리 상태에서 다음 rate의 합에 dt를 곱한다. 낙상 벌점 -20은 dt를 곱하지 않고 한 번 적용한다.

- 목표 경로 접선 속도 0.5m/s: `2 * (exp(-((v_parallel-.5)/.25)^2) - exp(-4))`. 정지에는 이 항의 보상 0.
- 실제 진행: `2 * clip(progress / dt, -1, 1)`. 지그재그는 전진 x(m), 원형은 중심 (0,1) 기준 감긴 각도(rad), 반지름1이므로 호길이(m)와 동일.
- 경로 이탈: `-2 * min((centerline_error/.15)^2, 4)`. 지그재그는 중심선까지 y오차, 원형은 반지름 오차.
- 자세: `-.5 * (1-upright)`, 경로 방향 불일치 `-.2 * (1-cos_heading)`.
- 낮은 몸통: 시작 높이의 70% 아래 부족분/시작높이의 제곱에 -2.
- 출발 0.5초 후 접선 속도가 0.05m/s보다 작으면 -0.5. 역주행에도 적용.
- 목표관절각 변화: `-.01 * mean((delta_target/.25)^2)`.
- 토크: `-.005 * mean((actuator_force/33.5)^2)`.

테스트는 정상 진행 > 정지 > 역주행, 경로 이탈·저자세·급격한 행동·큰 토크 감점, 출발 유예시간, 기존 보상 및 PPO 갱신을 확인한다. 합성 사례에서의 보상 순위가 실제 보행 학습 성공을 보장하지는 않는다.

각 task당 1000 PPO update = 768,000 transitions를 학습한다. 기존 v1은 100 update였으므로, 100 update checkpoint도 별도 보존·평가해 같은 예산의 보상 변경 효과와 10배 학습 효과를 구분한다. 최종 정책 선택은 평가 결과에 따르지 않는다. 지그재그·원형 정책은 별도 학습한다.

평가 표의 보상은 변경하지 않은 기존 task_reward이다. 새 학습 보상 합은 별도로 저장한다. 보상 스케일이 다르므로 구/신 학습 보상의 크기를 직접 비교하지 않는다. 진행·낙상·경로 이탈·몸통 높이도 확인한다.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl transformer-transformer/.venv/bin/python -u codesign/code/tt_dynamics/train_task_rl.py --task zigzag --reward-version dense --updates 1000 --out codesign/runs/pipeline_v1/task_rl_v2/zigzag_seed0
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl transformer-transformer/.venv/bin/python -u codesign/code/tt_dynamics/train_task_rl.py --task circle --reward-version dense --updates 1000 --out codesign/runs/pipeline_v1/task_rl_v2/circle_seed0
```

한계: CPU MuJoCo, 단일 seed, 고정 생성 E, 상태 S는 물리 결과. E 최적화나 E·S·A diffusion 공동 생성 통합을 수행하지 않는다. 4초 평가는 장기 보행이나 완주를 입증하지 않는다.
