# Pipeline v1 — e000 rollout 수집 계획 (2026-10-17)

파이프라인 4단계 (remind.md §2). 목적 = 전체 파이프라인 작동 확인이므로 형상은 **e000 하나**만 쓴다. 여러 형상 수집은 파이프라인 확인 후.

## 결정 사항
- **전환 정책은 따로 학습하지 않는다.** TT 는 dynamics 를 배우고 최종 실행은 생성한 A 의 MJX open-loop 이므로 expert 간 hand-off 가 필요 없다. 필요한 것은 전환 구간의 **데이터**.
- 수집 = DeepMimic composite 방식. 고정된 primitive expert 19 개를 무작위로 이어 10 s sequence 를 실행한다 (TT 10 s horizon, Q11).
- 낙상도 실제 dynamics 이므로 버리지 않고 `fell`·`valid` 로 표시해 저장한다.
- 보류: 전환 없는 데이터 vs 전환 포함 데이터로 학습한 TT 의 전환 구간 예측 오차 비교 (6 단계에서).

## 실험 조건
| 항목 | 값 |
|---|---|
| 코드 | `code/motion_tracking/collect_rollouts.py` |
| 형상 | e000 (topology `00000011`, 연속 치수 중간값), 토크 한계 ±33.5 N·m |
| expert | `runs/pipeline_v1/experts/*__r00` 19 개, 각 폴더 최신 offline-run 의 마지막 checkpoint (5,253,120) |
| reference | `runs/pipeline_v1/retargeting/e000/00000011/<모션>.npz` |
| env | eval env: kick 0, 초기 교란 0, obs noise = 학습 cfg 그대로, 오차 종료 끔 (낙상만 판정) |
| 길이 | 10 s = 500 step (50 Hz) |
| 전환 | 1~3 s 마다 무작위 다른 모션. 진입 프레임 = 관절각·접지 최근접 (`switch_entry=match`), B reference 를 로봇 xy·yaw 에 정렬. 클립 끝이면 즉시 전환 |
| 시작 | 무작위 모션·무작위 프레임 (RSI) |
| 저장 | `runs/pipeline_v1/rollouts/e000/seq_seed<N>.npz` + `_pairs.csv` (전환 pair 별 1 s 내 낙상) |

npz 키: `qpos [B,501,19]`, `qvel [B,501,18]`, `contact [B,501,4]`, `action [B,500,12]` (정책 출력), `target_q [B,500,12]` (PD 목표 관절각),
`clip`·`ref_frame`·`switched`·`valid [B,500]`, `fell [B]`, `E [13]`, `motions [19]`, `dt`.

## 이 PC (V100) 시험 결과
- 1 모션 수집 (squat, trotBackwardFast, 8 ep): 관절 오차 1.4~2.1° — 기존 평가값 (2.01°, 1.47°) 과 일치, obs·프레임 정렬 정상.
- 10 s 혼합 (모션 4 개, 8 ep): 7/8 이 10 s 완주, 전환 4.9 회/ep, 전환 직후 0.5 s 관절 오차 2~6° (단독 실행과 비슷).
- 낙상 1 건 = squat → trotForwardSlow 직후 (같은 pair 다른 ep 는 성공 → 전환 시점 자세 의존).

## 실행 순서 (다른 PC)
1. `git pull` 후 `codesign/` 에서 env 설정:
   `export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=offline`
2. **smoke** (19 expert, 8 ep) — 오류 없음 + 소요 시간 측정:
   `../transformer-transformer/.venv/bin/python code/motion_tracking/collect_rollouts.py runs/pipeline_v1/experts/*__r00 --design e000 --episodes 8 --out runs/pipeline_v1/rollouts_smoke`
3. **본 수집** (tmux, 종료 알림): seed 0 → 1 → 2 → 3, 각 256 ep (총 1,024 ep ≈ 5,000 전환, pair 342 개당 평균 ~15 회).
   `for s in 0 1 2 3; do .../python code/motion_tracking/collect_rollouts.py runs/pipeline_v1/experts/*__r00 --design e000 --episodes 256 --seed $s || exit 1; done`
   256 ep 가 GPU 메모리를 넘으면 `--episodes 128` 로 seed 수를 2 배로.
4. 결과를 seed 별 출력 1 줄 + `_pairs.csv` 상위 낙상 pair 로 기록 (remind.md, 항목당 3 줄).

## 판정 (seed 0 결과로)
| 분기 | 조건 | 다음 |
|---|---|---|
| A | 전체 낙상 < 20 % | seed 1~3 진행 → 5 단계 (TT dataset) |
| B | 특정 pair 에 낙상 집중 (pair 낙상 > 50 %, 전환 ≥ 5 회) | 데이터는 유지, pair 목록 기록. 제외 여부는 사용자 결정 (`switch_forbid`) |
| C | 전체 낙상 ≥ 20 % 또는 모션 구간 낙상 다수 | 수집 중단, 진입 프레임 매칭·xy·yaw 정렬·obs noise 부터 점검 |

## 주의
- 이 PC 실측 학습 속도: 4070 SUPER 모션당 18.4 분, V100 42 분 (환경 32 개에서 4070 이 2.3 배 빠름). 수집도 다른 PC 권장.
- `data.contact` DeprecationWarning 은 무시해도 된다 (MJX 버전 경고).
- Issue #2 의 expert 중 11 개는 학습 말기 KL > 1 (최대 1.2e4). e000 평가는 19/19 통과라 수집에는 사용하지만, 낙상 많은 모션이 이 목록과 겹치는지 확인할 것.
- e003 (미학습 형상) 0/19 통과 — 다형상 수집 단계에서 학습 형상 수가 핵심 변수.
