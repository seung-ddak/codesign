# Task action RL replacement pilot

목표: v16 twin이 생성한 E는 고정하고, 행동 A를 reward-gradient guidance 대신 PPO로 생성한다. S는 실제 MuJoCo 실행으로 얻는다. E 학습과 diffusion S 공동 생성 통합은 이번 실험 범위 밖이다.

- 비교: 동일한 E·S0·CPU 물리 조건에서 기존 guidance 행동, 학습 전 home 정책, PPO 학습 후 정책.
- 과제: 지그재그, 원형. 각 과제의 기존 생성 형상 16개 중 0–11에서 학습, 12–15는 학습 제외 평가.
- 관측: 현재 S42, E13, 이전 목표관절각12, 정규화 시간1. 정책은 home 대비 목표관절각12를 출력. 모터 kp30, 토크 ±33.5Nm, 제어 50Hz, 시뮬레이션 200Hz.
- 학습: 저장소의 `quad/rsl_rl` PPO, 64×64 MLP, seed0, 100 updates × 64 steps × 12 envs = 76,800 transitions/과제.
- 학습 보상: 기존 task reward의 prefix 차분. 조기 낙상 유도를 막기 위해 낙상 시 -20 추가. 평가에는 추가 낙상 벌점을 넣지 않으며 원래 task reward를 사용한다.
- horizon: 기존 보고서와 동일한 200 상태 프레임/199 제어 전이(약 4초).
- GPU 드라이버가 현재 세션에서 작동하지 않아 CPU 실험으로 실행. 기존 MJX 수치와 직접 동일시하지 않는다.
- 첫 두 시도는 Adam 초기화 시 설치된 Triton native extension segfault로 중단. 학습 update가 수행되지 않았으며 debug 폴더/로그에 보존. 현재 CPU 전용 스크립트에서 Triton import를 제외했다.

저장소 루트에서 실행 (출력 디렉터리는 새 경로여야 함):

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl transformer-transformer/.venv/bin/python -u codesign/code/tt_dynamics/train_task_rl.py --task zigzag --updates 100 --out codesign/runs/pipeline_v1/task_rl_v1/zigzag_seed0_retry
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl transformer-transformer/.venv/bin/python -u codesign/code/tt_dynamics/train_task_rl.py --task circle --updates 100 --out codesign/runs/pipeline_v1/task_rl_v1/circle_seed0
```

`protocol.json`에 코드/입력 해시와 조건, `training.jsonl`에 update별 지표, `policy.pt`에 최종 정책, `summary.json`에 후보별 평가, NPZ에 실제 S·A·qpos를 저장한다. 결정론적 정책의 초기 상태는 고정되어 있으므로 같은 조건의 반복 에피소드를 독립 표본으로 세지 않는다. 최종 정책을 평가하며 평가 보상으로 checkpoint를 고르지 않는다.

한계: 단일 seed·짧은 학습 예산의 pilot. 4개 학습 제외 형상은 같은 생성 모델에서 나온 후보이며 완전히 새로운 형상 분포에 대한 일반화를 입증하지 않는다. 보상 개선만으로 task 성공을 선언하지 않고 진행 방향·낙상도 확인한다. E가 task에 유리하게 생성되는 문제는 별도 단계가 필요하다.
