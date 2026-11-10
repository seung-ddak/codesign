# V100 shared PPO 파일럿 결과 (2026-11-05)

사용자가 V100 서버에서 직접 실행한 파일럿 결과의 원본 보관본입니다.

- 원본 압축파일: [v100_pilot_review.tar.gz](v100_pilot_review.tar.gz)
- Git blob SHA-1: `183d36f8f1cbdeec56949d1b01911c131ad4cede`
- 파일 SHA-256: `9c821e5a3bb5c7424f3184a0aa26ed48a0335edf1f565b3ecc213191759f4b06`
- 압축파일 크기: 1,428,613 bytes

## 파일럿 결과 (압축파일 내부 state.json / audit.json 기준)

| 지표 | 결과 |
| --- | ---: |
| 실행 상태 | complete |
| PPO 누적 학습 전이 | 2,048 |
| 수집 Episode | 16 / 16 |
| TT 사용 가능 Episode | 16 |
| 유효 Dynamics 전이 | 3,612 |
| 낙상 | 6 / 16 |
| 원본 모션 커버리지 | 4 / 4 |
| 형상·모션 쌍 | 8 / 8 |
| 2개 시작 조건 수집 쌍 | 8 / 8 |
| 원본 시간 구간 커버리지 | 94.56% |
| 형식 오류 | 0 |
| 중복 초기 상태 | 0 |

이 결과는 **실행·수집 경로의 확인**입니다. PPO 수렴이나 정밀 모션 추종 성공을 입증하는 결과가 아닙니다.

### 압축파일 구성

`state.json`, `audit.json`, `metrics.jsonl`, `training_coverage.json`, `checkpoint.json`, `report.html`, `logs/`, `rollouts/`, `evaluation/`.

학습 가중치와 optimizer 전체 체크포인트는 이 압축파일에 **포함되지 않습니다**. `checkpoint.json`은 서버 원본 위치를 가리킵니다.

### 압축 해제 예시

```bash
mkdir -p runs/shared_calm_v100_pilot_review
 tar -xzf runs/v100_pilot/v100_pilot_review.tar.gz -C runs/shared_calm_v100_pilot_review
```
