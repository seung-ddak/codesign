# T8 TaskEnv π_θ

[Workflow](./t8_task_env.html) ([아티팩트](https://claude.ai/artifact/C6sn9btvjUmm1dnUZ3Mp9z)) — legged_gym rough terrain 위에서 보행·회전·점프·계단을 정책 하나로 학습하고, 동결한 checkpoint 를 시나리오 5 개로 평가해 보고서를 만든다 (D63, Q15 B · Q16 a · Q17 b).

## 구현

- `stage2/code/terrain.py`: legged_gym `Terrain.curiculum()` 재구현 (10 × 20 타일, 5 종 비율, 경계 25 m). `python terrain.py` = 자체 검사
- `stage2/code/task_env.py`: `TaskEnv(LocomotionEnv)` — heightfield, 높이 측정 187 점, 지형 커리큘럼, 점프 command·보상, obs 257
- `stage2/code/multi_design_env.py`: `env_cls` 선택, 형상 간 heightfield 1 벌 공유 (복사하면 GPU 수십 GB)
- `stage2/code/train_locomotion_policy.py --env task`: brax PPO (T5 와 같은 설정)
- `stage2/code/eval_task_policy.py`: 시나리오 5 개 (평지 전진·평지 좌회전·점프·계단 오르기·내리기) × 전 형상, DR 끔, `--videos`, `--reuse`
- `stage2/code/make_stage_report.py t8`: `report.html`
- `stage2/code/test_task_env.py`: obs·스폰·지면 접촉·높이 측정·점프·커리큘럼·heightfield 공유 검사

## 원본과 다른 점

- 점프 (legged_gym 에 없음, D36 예외): 확률 0.2, `jump_air` 1.0 · `jump_height` 5.0 (× dt), 점프 중 `lin_vel_z` 페널티 끔. scale 은 임시값
- MJX 는 cylinder–heightfield 충돌이 없어 thigh·calf 접촉용 원통을 같은 크기 캡슐로 바꿈 (컴파일 뒤라 질량·관성 불변)
- brax auto-reset 은 끝난 에피소드를 못 보므로 커리큘럼 단계를 `AutoResetWrapper_preserve_info` 에 두고 다음 에피소드 첫 step 에 해당 타일로 옮김 (첫 step 의 높이 측정은 reset 타일 기준)

## 실행

```bash
P=../transformer-transformer/.venv/bin/python; S=../scripts/tmux_run.sh
MUJOCO_GL=egl $P stage2/code/test_task_env.py
TMUX_RUN_CWD=$PWD bash $S T8_pi_smoke "$P stage2/code/train_locomotion_policy.py --env task --out stage2/runs/T8/pi_smoke --seeds-per-choice 1 --num-envs 512 --timesteps 1228800 --checkpoints 2"
TMUX_RUN_CWD=$PWD bash $S T8_pi_smoke_eval "$P stage2/code/eval_task_policy.py --run stage2/runs/T8/pi_smoke --level 3 --videos"
$P stage2/code/make_stage_report.py t8 --run stage2/runs/T8/pi_smoke --title "T8 π_θ smoke"
```

컴파일이 학습 약 41 분, 평가 약 37 분 걸린다 (16 구조 그룹 × heightfield 충돌).
