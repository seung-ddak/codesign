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
   `../transformer-transformer/.venv/bin/python code/motion_tracking/collect_rollouts.py runs/pipeline_v1/experts/*__r00 --design e000 --episodes 8 --out runs/pipeline_v1/rollouts_archive/smoke`
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

## 4070 SUPER 실행 결과 (2026-10-17)
- 19 expert smoke: 8 ep, 3 완주 / 5 낙상(62.5%), 유효 2,610 step, 전환 30회. `rollouts_archive/smoke/e000/seq_seed0.npz`.
- seed 0: 256 ep, 150 완주 / 106 낙상(41.4%), 유효 104,465 step, 평균 8.16 s, 전환 1,208회 / 336 pair. 전환 후 1 s 내 낙상 98회, 그 외 8회. **분기 C: seed 1~3 중단**, 데이터 보존.
- 전환 ≥5회·낙상 >50% pair: tiltRoll→trotCCWSlow 4/5, walkCCWSlow→walkForwardFast 6/8. pair 제외는 적용하지 않았다. 전체 표는 `rollouts/e000/seq_seed0_pairs.csv`.
- 검증: `check_rollouts.py`로 state/action 시간축, finite 값, valid 마스크, 모션 전환 연결, PD action 변환 검사 통과. 19 expert의 eval 설정은 clips/reference 경로 외 일치; kick·초기 교란 0. 낙상 원인 확정 및 수정 후 재수집은 아직 하지 않았다.

### 저장 단위와 재현 정보
- NPZ 하나는 seed 하나의 병렬 episode 묶음이다. seed 0은 state `[256,501,...]`, action `[256,500,12]`. 50 Hz, 최대 10초이며 낙상 후 구간은 `valid=False`다. 전체 배치가 조기 종료되면 시간축도 짧아진다.
- `action`은 정책 출력, `target_q`는 action 제한 후 실제 PD 목표다. `next_clip`은 step 직후 모션 번호다. `checkpoint_paths`, `action_offset`, `action_scale`, `clip_actions`를 본 수집 NPZ에 추가했다. smoke는 메타데이터 추가 전 실행본이다.
- 체크포인트를 새로 학습·복사하지 않는다. 기존 expert 체크포인트를 읽는다. `_pairs.csv`는 seed 전체의 전환 집계이며 `.log`는 실행 로그다. 영상은 수집하지 않는다. 동일 seed 출력이 있으면 덮어쓰지 않고 오류로 종료한다.

## 다형상 추가 수집 (2026-10-17 사용자 요청)
- 기존 e000 mixed seed 0은 보존한다. e000/e001/e002 각각 primitive 304 episode(19 모션 × 16회), e001/e002 mixed 256 episode씩 추가한다. 미학습 e003은 이번 수집에서 제외한다.
- primitive는 모션 시작 프레임부터 클립 끝 또는 낙상까지 실행한다. 모션은 episode별 순환 배정하고 전환은 없다. 초기 물리 교란은 0이며 관측 noise는 기존 학습값이다. 반복은 독립적인 새 모션 reference를 뜻하지 않는다.
- mixed는 기존과 동일한 10 s 무작위 전환이다. 사용자와 확인한 목적에 따라 낙상률만으로 중단하지 않고 실패도 보존한다. 수치 오류 또는 파일 검증 실패 시 중단한다.
- `collect_rollouts_batch.sh`: 각 묶음 저장 → `check_rollouts.py` 검증 → 해당 NPZ·CSV·실행 로그만 commit/push. push 실패 시 중단하며 강제 push하지 않는다. 이미 있는 출력은 덮어쓰지 않는다.
- 출력은 `rollouts/<design>/primitive_seed0.npz`와 `seq_seed0.npz`. `mode`, `seed`, `clip_start`, `clip_end`를 추가하여 정상 클립 종료와 낙상을 구분한다. 시퀀스 i의 유효 길이는 `valid[i].sum()`이며 인접 상태 쌍과 action을 함께 사용한다.
