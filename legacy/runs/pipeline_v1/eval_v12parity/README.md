# eval_v12parity: v12 동일 평가 + remind 8-43 사전 기준 판정 (학습 없음)

- 결과: run 폴더에 `gate.json` (기준 1·2·3 통과 여부), `evaluation/comparison.json`, `report.html`. 상태 `eval_status.json` (학습 status.json 과 분리).
- 기준 (8-43, v12 Mode 1 평가 동일): 틀린 A 추가 오차 (속도) ≥ 0.01 · E 교체 시 예측 전진 변화 부호 일치 ≥ 70 % (고정 E 형상 평균 쌍) · 공동 생성 생존 ≥ v12 (7/16 · 10/16).
- 실행: `python runs/pipeline_v1/eval_v12parity/run_eval.py runs/pipeline_v1/<run>` (tt_joint/checkpoint.pt, conditions.json 필요). v12 `run_comparison.py` 의 train 이후 단계를 그대로 옮기고 `held_mode1` (`--pin-mode 1`) 을 추가했다. 모델 비교 = v8 / v9 / v12 / 대상.

검증: v12 fixedE summary 로 `sign_agreement` 를 돌리면 8-38 기록 (zigzag 36/64, circle 35/64, 낙상 17) 과 일치. v12 형상 쌍 부호 일치율 0.82 / 0.61.
