# PD reference rollout로 학습한 TT와 기존 PPO 데이터 모델 비교

**판정: PD 분포의 예측은 개선됐지만, 기존 미학습 형상 평가와 Co-Design Gate는 미달.**

[비교 보고서·그래프·실제 MJX 영상 32개](report.html)

60k 학습과 176개 실제 MJX 실행을 완료했다. PD 검증의 높이·기울기·속도 움직임 상관은 v8의 0.123/0.635/0.182에서 0.339/0.784/0.369로 개선됐다. 기존 시험 형상 516쌍에서는 개선 기준을 통과하지 못했다. 2초 경로 차 중앙값은 12.2→9.6cm이며, 4초 공동 생성 생존은 zigzag 16/16·circle 15/16이다. 형상 평균 보상의 예측·실제 순위 상관은 0.159/0.053으로 기준 0.6에 미달했다. 생존과 Task 목표 달성을 구분한다.

결과 보고서는 `report.html`, 단계별 진행 상태는 `status.json`에 저장한다.

- 입력: `../pd_rollouts/e*/primitive_seed0.npz` 16개. PPO/E-swap 데이터를 학습에 섞지 않는다.
- 주 기준: 같은 16형상을 학습한 v8 A. 48형상 v9는 보조 기준이다.
- 학습: v8과 같은 FiLM, horizon 32, 60k step, batch 32, Mode 2/4, diffusion 5, minmax, DDIM, seed 0.
- 기존 평가: v9 checkpoint가 정한 동일 source split, 동일 시험 형상 8개와 516쌍, seed 8개, v8 오차 단위. 학습 형상 평가도 함께 실행한다.
- PD 평가: PD 모델의 미학습 시작점 15그룹을 세 모델 모두에 동일하게 적용한다. 형상별 리타겟팅 A가 달라 동일-A 형상 효과로 해석하지 않는다.
- 실제 MJX: 2초 smoke 16후보, zigzag/circle 각각 4초·고정 16형상 × 후보 4개·공동 생성 16개. 기존과 같은 E/S0 처리와 scale을 유지한다.
- 모든 원본 데이터, checkpoint, 코드 해시는 `conditions.json`에 기록한다. 기존 v8/v9 물리 결과는 해시 일치를 확인한 뒤 재사용한다.

실행 진입점(저장소 루트 기준):

```bash
../transformer-transformer/.venv/bin/python -u runs/pipeline_v1/fast_mode24_v10_pd_reference/run_comparison.py
```

진행 중인 작업에 위 명령을 중복 실행하지 않는다. 완료된 단계는 `status.json`을 기준으로 건너뛴다. 학습은 중간 checkpoint를 저장하지 않는 기존 학습기를 유지하므로, 중단된 학습은 다시 시작해야 한다.

자원: CPU 2코어, torch GPU allocator 60%(평가 25%), JAX 풀 40%, 메모리 선할당 끔. 가용 RAM 4GiB·GPU 여유 1GiB·루트 디스크 1GiB·작업 디스크 8GiB 아래가 두 번 연속 관측되면 이 작업의 프로세스 그룹만 중단한다. 로그는 `resources.jsonl`, 임시파일·캐시는 `/home`의 `.runtime`에 둔다.

해석 범위: 수집 제어 방식과 데이터 양·초기 상태·행동 분포가 함께 다르므로 제어기 하나의 인과 효과는 분리하지 않는다. PD 분할은 같은 모션의 다른 시작점을 학습에 허용하므로 미학습 모션 평가는 아니다. Mode 2에서 A를 고정하는 기존 진단을 유지하며, Mode 1을 학습한 조건부 모델로 해석하지 않는다. 최종 10초 평가는 포함하지 않는다.

## 다른 PC에서 이어받기

`cleanup/shared-ppo` 브랜치를 내려받은 뒤 저장소 루트의 `SETUP.md`에 따라 기존 `portability` 묶음을 복원하고 `tt_local.patch`를 적용한다. 외부 TT의 base_env·model/utils·시각화 코드는 기존 묶음의 SHA-256과 일치하고, 추가 수정 3파일은 해당 패치와 일치함을 확인했다. CALM이 없는 현재 PC에서 기존 전체 묶음을 재생성하지 않았다.

PD 원본 16개, PD·v9 checkpoint, 평가 NPZ/JSON, 그래프, 영상, 실행 로그를 일반 Git 파일로 포함했다. 가상환경·자격증명·컴파일 캐시·재생성 가능한 `.mjb`는 제외했다. `conditions.json`의 해시는 원본 PC의 측정 당시 파일 기준이므로 새 PC에서 경로를 rebase하면 경로가 들어간 파일의 해시는 달라질 수 있다.

측정과 보고서는 이미 완료되어 있으며 `report.html`을 열어 결과를 확인할 수 있다. 원본 경로가 들어 있는 checkpoint의 평가용 데이터 경로는 새 PC에 맞게 확인해야 한다. 다른 PC에서 장시간 GPU 실험을 다시 실행한 결과까지 검증한 것은 아니다.
