# Jump Expert

Locomotion Expert(T5 → 257 obs) 가중치에서 분기해 제자리 점프·착지를 학습한다. 보행 command 는 0 고정.

## 사용 파일
| 파일 | 역할 |
|---|---|
| `code/train_policy.py` | 학습 실행 (`--phase jump`). 시작 checkpoint = `runs/expert/locomotion/checkpoints/000000000000` |
| `code/task_env.py` → `FlatJumpEnv` | 점프 스케줄 (안정 45 → [점프 80 → 대기 10] 반복), 목표 높이 0.6 m, 착지 보상 |
| `stage2/code/task_env.py` 상단 설정 블록 | 보상 scale (`jump_air`, `jump_height`, tracking·penalty), episode 20 s, noise·push |
| `stage2/code/locomotion_env.py` | 보상 식, reset, action_scale |
| `code/record_jump_rollout.py` | 학습 후 영상 + per-step trace (`<run>/videos/`) |

## 실행 (RL 폴더에서)
```bash
bash codesign/code/run_jump_smoke.sh <run>   # smoke: 아래 긴 명령과 같음

bash scripts/tmux_run.sh jump_<run> "bash -c 'P=transformer-transformer/.venv/bin/python; \
MUJOCO_GL=egl \$P codesign/code/train_policy.py --out codesign/runs/expert/jump/<run> --phase jump \
--seeds-per-choice 1 --timesteps 153600 --num-envs 64 --checkpoints 2 --learning-rate 3e-5 --max-groups 1 --seed 0 && \
MUJOCO_GL=egl \$P codesign/code/record_jump_rollout.py --run codesign/runs/expert/jump/<run>'"
```
위 인자는 smoke 조건. 본 학습은 `--seeds-per-choice`·`--max-groups`·`--timesteps`·`--num-envs` 를 키운다.

## 결과 확인
- `<run>/progress.jsonl`: `step` = 전체 env step 누적 (64 env × 24 step 단위), 1 step = 0.02 s
- 점프 지표: `episode/jump_air`, `episode/jump_height`, `episode/jump_takeoff_success` (점프 인정), `episode/jump_landing_success` (착지 성공)
- `<run>/videos/jump_rank_0_trace.json`: `max_base_z_m`, `max_base_rise_m` (step 45 이후, 스폰 낙하 제외)

## Run
