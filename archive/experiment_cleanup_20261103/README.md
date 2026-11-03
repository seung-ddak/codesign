# 2026-11-03 실험 파일 정리

현재 CALM 10,000개 수집과 E/S/A 공동 생성 코드가 사용하는 입력은 보존한다. 과거 실험을 쓸모없다고 단정해 폐기하지 않고, 종료된 초기 시도와 중단된 실행은 아래 압축본에 원래 경로 그대로 보관한다.

## 보관 대상

| 묶음 | 원래 위치 | 이유 |
|---|---|---|
| compiled_models | runs/**/designs/*.mjb | 형상 JSON과 함께 남아 있던 컴파일 사본. 읽는 실행 코드는 없고 생성 코드만 확인됨. 원본 바이너리도 압축 보존 |
| fast_mode24_v4_warmup_s0 | runs/pipeline_v1/fast_mode24_v4_warmup_s0 | 종료된 초기 자세 시험; 외부 실행 코드에서 입력으로 참조하지 않음 |
| v7_followup_ablation | runs/pipeline_v1/v7_followup_ablation | 과거 추가 실험; 측정값·정책·보고서·영상까지 압축 보존 |
| rl_denoising_100iter | runs/pipeline_v1/rl_denoising_100iter | 20초 설정에서 중단된 실행 |
| rl_denoising_5s_100iter | runs/pipeline_v1/rl_denoising_5s_100iter | 61 iter 부근에서 중단된 과거 실행. 완료 실험으로 표시하지 않음 |
| calm_collection_pilots | runs/calm_composed_10000/pilot*, 시험 로그 | 본 수집 전 종료한 수행·재개 검증; 현재 dataset과 collect.log는 보존 |

모든 압축본은 분할당 최대90 MiB다. `manifest.json`은 원본 파일별 SHA-256, 원래 경로, 분할본별 SHA-256, 검증·정리 완료 상태를 기록한다. 원본을 지우기 전에 압축 파일을 다시 읽어 모든 파일의 해시를 확인했다. Python·pytest·JAX 캐시는 재생성 가능하므로 압축하지 않았다.

## 정리 결과

- 원본 파일 2,642개를 해시 검증한 압축본으로 보관했다.
- 압축 전 8.54GB → 압축 후 0.50GB.
- 재생성 가능한 캐시 7,026개, 52.5MB를 정리했다.
- 총 확보 공간 약8.10GB. 현재 dataset 증가량과는 별도로 계산했다.
- 2026-11-03 커밋에 압축본을 포함하고 portability 스냅샷을 갱신했다. 압축 보관한 실험은 명시적으로 복원할 때만 다시 생긴다.

## 복원

codesign 폴더에서 필요한 묶음만 복원한다. 같은 경로에 새 실험을 진행했다면 먼저 경로를 확인한다. 아래 명령은 기존 파일을 덮어쓰지 않는다.

```bash
cat archive/experiment_cleanup_20261103/rl_denoising_5s_100iter.tar.gz.part* | tar -xzk -C .
```

다른 묶음은 파일명만 바꾼다. 컴파일 모델 하나만 복원하려면:

```bash
cat archive/experiment_cleanup_20261103/compiled_models.tar.gz.part* | tar -xzk -C . runs/pipeline_v1/task_rl_v1/designs/원래파일명.mjb
```

정확한 파일명은 manifest.json을 확인한다. 복원 전 목록은 `cat ...part* | tar -tz`로 볼 수 있다. 보관한 실험의 보고서·영상은 해당 묶음을 원래 경로에 복원한 뒤 연다. 과거 문서의 실행 명령과 경로는 재현 기록으로 유지한다.

## 점검 결과

What this repo does: CALM 모션 수집, 형상·상태·행동 생성, 물리 시뮬레이션 평가를 수행한다. 한 명의 연구자가 여러 실험을 실행하는 사용량을 기준으로 실험 파일 수명과 의존 관계를 점검했다.

### Should fix

1. **컴파일 모델이 중복 공간을 차지함** (`runs/**/designs/*.mjb`)
   - What this is: 생성 형상의 컴파일된 MuJoCo 모델이다.
   - Problem: JSON에 재생성 방법이 있고 코드에서 읽지 않는 바이너리 1,664개가 약8.41GB를 차지했다.
   - Fix: 원본 해시를 검증한 압축본으로 보관하고 작업 폴더의 사본을 정리했다.
   - If we skip it: 실험마다 같은 종류의 파일이 계속 쌓인다.

2. **중단된 학습이 진행 중으로 표시됨** (`runs/EXPERIMENT_INDEX.md`)
   - What this is: 실험 탐색용 목록이다.
   - Problem: 5초 100iter 실행이 중단됐는데도 진행 중 링크가 남아 있었다.
   - Fix: 중단된 결과를 압축 보관하고 목록을 보관·복원 안내로 갱신한다.
   - If we skip it: 완료 여부와 최신 실행을 잘못 판단한다.

3. **초기 검증 결과와 본 수집 데이터가 섞여 있음** (`runs/calm_composed_10000`)
   - What this is: 본 수집과 세 개의 작은 검증 출력 폴더다.
   - Problem: 탐색기에서 시험 데이터와 실제 10,000개 수집 데이터가 함께 나타났다.
   - Fix: 시험 폴더·로그를 압축 보관하고 dataset·collect.log·README만 남긴다.
   - If we skip it: 시험 결과를 본 수집 결과로 오인하기 쉽다.

Verdict: 필요한 과거 실험은 보존하고, 종료된 시험·컴파일 사본·캐시를 정리한다.
Lean: 실행 코드와 의존성은 변경하지 않았다.
Not checked: 전체 저장소의 알고리즘·보안·물리 정확성 감사는 수행하지 않았다. 모든 과거 실험을 재실행하지 않았으며, 참조되는 체크포인트·학습 데이터·보고서 영상은 정리 대상에서 제외했다. 이후 사용자 요청에 따라 Git 커밋·푸시를 진행하며 portability 스냅샷도 갱신한다.
