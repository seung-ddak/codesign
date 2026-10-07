# Locomotion Expert

T5 try1 (`stage2/runs/T5/try1/checkpoints/000024576000`, 69 obs) 를 257 obs 로 이식한 동결 정책. 학습하지 않는다.

## 체크포인트
- `checkpoints/000000000000`: 257 obs (T5 69 + jump flag 1 + height scan 187). 신규 입력 첫 layer 가중치 = 0, obs normalization mean 0 / var 1 → 이식 직후 T5 와 같은 action
- 다른 expert (`jump` 등) 의 시작 checkpoint 로 쓰인다 (`code/train_policy.py` 기본값)
- 이식 스크립트는 저장소에 없다 (다른 PC 에서 생성)

## 사용 파일
| 파일 | 역할 |
|---|---|
| `code/task_env.py` → `FlatTaskEnv` | T5 관측 + 신규 slot 0 패딩 (257) |
| `code/train_policy.py --verify-only` | checkpoint 로드·관측 차원 확인만 |
| `stage2/code/eval_locomotion_policy.py` | 고정 command 7 개 평가 (T5 기준선) |

## 실행 (RL 폴더에서)
```bash
transformer-transformer/.venv/bin/python codesign/code/train_policy.py --verify-only
```
