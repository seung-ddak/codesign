# 2026-11-03 추가 코드·smoke 정리

What this repo does: CALM 연속 모션 수집과 TT의 E/S/A 생성·물리 평가를 수행한다. 한 명의 연구자가 실행하는 실험 저장소를 기준으로, 종료된 도구와 임시 출력의 사용 관계를 점검했다.

사용자의 실제 정리 요청에 따라 감사 결과를 적용했다. 스킬의 기본 읽기 전용 감사 범위보다 사용자의 정리 요청을 우선했다.

## 결과

- 실행 코드·전용 테스트·쉘 스크립트 **10개, 914줄**을 작업 트리에서 보관본으로 옮겼다.
- 전체 **208개 파일, 77개 디렉터리**를 압축 보관했다.
- 원본 175.7MB → 압축본 51.3MB, 약124.3MB 확보.
- 원본 파일별 해시와 압축본을 다시 읽은 해시가 일치한 뒤 원본을 정리했다. 정확한 경로·해시는 [manifest.json](manifest.json)에 있다.

## Should fix

1. **보관한 v7 실험의 실행 코드가 현행 코드와 섞여 있음** (`code/tt_dynamics/run_v7_ablations.py`, `eval_v7_ddim100.py`)
   - What this is: 과거 v7 추가 실험·100단계 DDIM 평가를 위한 실행기와 평가기다.
   - Problem: 결과는 앞선 정리에서 보관했지만 전용 실행 코드, 쉘 진입점, 두 테스트 파일, 실험 설명은 남아 있었다. 호출은 이 묶음 안에서만 확인됐다.
   - Fix: 실행기·평가기·호출 쉘·전용 테스트·설명을 `retired_code`에 함께 보관했다.
   - If we skip it: 현재 E/S/A PPO 실험과 과거 v7 실행 경로를 혼동한다.

2. **일회성 스크립트가 현행 실행 경로에 남아 있음** (`run_task.sh`, `check_track_env.py`, `fast_mode24_v5_smoke*.sh`, `.runtime/old_cond_masks.py`)
   - What this is: 자동 커밋·main 푸시 래퍼, 특정 클립 20step 수동 검사, 옛 PC 경로를 고정한 v5 smoke, 과거 함수 사본이다.
   - Problem: 현행 코드에서 호출되지 않는다. v5 스크립트는 `/mnt/gstore/...`를 고정하고, old_cond_masks는 독립 실행에 필요한 전역 변수가 없는 사본이다.
   - Fix: 원본을 `retired_code`에 보관했다. 공용 `check_fast_smoke.py`, `check_multi_design.py`, 회귀 검증은 유지했다.
   - If we skip it: 종료된 수동 시험과 현재 사용할 진입점을 구분하기 어렵다.

3. **시험 출력이 본 실험과 같은 폴더에 쌓임** (`.runtime/*smoke*`, `experts/smoke`, `rollouts_archive/*smoke*`)
   - What this is: 짧은 초기 실행, RL 디버그, 시험 체크포인트와 임시 진단 출력이다.
   - Problem: 현재 수집·학습 입력으로 사용되지 않는 출력이 흩어져 있다.
   - Fix: `.runtime` 시험 출력은 `runtime_smoke`, expert·rollout·RL 시험은 `unused_smoke_results`로 보관했다.
   - If we skip it: 시험용 체크포인트와 본 학습 결과를 혼동한다.

## 검증

- 남은 전체 테스트: **150 passed, 3 warnings, 9.46s** (CPU).
- 제거된 코드의 외부 실행 참조 없음: 나머지 code/tests/rl_scripts/runs/.github의 Python·쉘·YAML 검사.
- 코드 묶음을 임시 폴더에 실제 복원하고 파일별 SHA-256 재검증 통과.
- 두 차례 정리에서 Git 추적 파일 삭제908개 모두 검증된 압축본에 포함됨.
- 변경한 탐색 문서의 로컬 링크 검사 통과.
- 실행 중인 CALM 수집의 두 소스 파일 해시가 collection_config.json과 일치함.

실행한 검증 명령:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/codesign-cleanup-mpl ../transformer-transformer/.venv/bin/python -m pytest -q -p no:cacheprovider tests --basetemp=/tmp/codesign-cleanup-pytest
```

## 보관 묶음과 복원

| 묶음 | 내용 |
|---|---|
| retired_code | v7 실행기·평가기·전용 테스트 2개·실행 쉘·문서, v5 smoke 쉘 2개, run_task.sh, check_track_env.py, old_cond_masks.py |
| runtime_smoke | freefix·tt_cgr smoke 및 로그, actual-share/group-loss smoke, v9 임시 진단, 일회성 보고서 보정·정리 스크립트 |
| unused_smoke_results | expert smoke, rollout smoke 2개, 초기 diffusion PPO smoke와 task PPO debug 출력 |

codesign 디렉터리에서 필요한 묶음만 복원한다. 기존 파일은 덮어쓰지 않는다.

```bash
cat archive/code_cleanup_20261103/retired_code.tar.gz.part* | tar -xzk -C .
```

v7 결과까지 필요하면 [앞선 정리](../experiment_cleanup_20261103/README.md)의 `v7_followup_ablation` 묶음도 복원한다. `run_task.sh`는 복원 후 실행 시 커밋·원격 푸시를 수행하는 과거 도구임에 주의한다.

Verdict: 현재 CALM 수집·RL 학습 경로는 유지하고, 종료된 코드와 출력만 보관했다.
Lean: -914 lines, -0 dependencies. 보관한 테스트는 제거한 v7 코드 전용 2개다.
Not checked: 전체 알고리즘·보안 감사나 과거 GPU 실험 재실행은 수행하지 않았다. 압축본은 이후 사용자 요청에 따른 커밋에 포함하며 portability 스냅샷도 갱신했다.
