# T6 평가기 검증

[다이어그램](./t6_evaluator.html)은 동결된 T5 정책을 미학습 형상에서 두 번 재평가하고 D56 기준으로 T7 진입 여부를 결정하는 흐름이다. T6는 학습이 아니며 정책과 형상을 갱신하지 않는다.

## 구현 위치

- `stage2/code/t6_evaluator_check.py`: 미학습 seed 선택, 반복 평가, 세 기준 판정
- `stage2/code/eval_locomotion_policy.py`: checkpoint 로딩과 고정 명령 7개 rollout
- `stage2/code/multi_design_env.py`: 128개 형상을 16개 구조 그룹으로 묶어 MJX 평가
- `stage2/runs/T6/try1/`: `designs.json`, `eval_rep1.json`, `eval_rep2.json`, `t6_result.json`

## 결과 해석

- `spearman < 0.8`: reset·종료·명령 할당 등 반복 평가 안정성 확인
- `generalisation_gap` 실패: T5 학습 형상 범위와 reward 재검토
- `choice_spearman`: 같은 choice 내부 continuous variation 분석용 진단값

기준값을 먼저 낮추지 않는다. 실패한 지표가 가리키는 코드만 확인한다.
