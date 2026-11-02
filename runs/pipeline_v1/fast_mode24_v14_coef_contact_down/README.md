# v14: v12 + 손실 고정 계수, 접지 그룹 ÷4 (remind 8-43 (1))

- 단변수: `--loss-balance coefficients --loss-group-weights 0.1653 0.1653 0.1653 0.1928 0.0138 0.2975`. 기존 loss 의 채널 합 (12/12/12/14/4, E 항 ≈ 0.4·raw_E — E 는 mode 4 (40 %) 에서만 미지) 에서 접지 4 → 1 만 바꿔 정규화한 값. v13 (실제 기여율 역수 가중) 과 다른 방식.
- 데이터·구조·Mode 1/2/4 (20/40/40)·seed·60k step = v12. 학습 `run_train.py` → 평가 `../eval_v12parity/run_eval.py` → `gate.json`, `report.html`.
- 한계: 그룹 평균은 미지·유효 항목으로 계산되므로 (Mode 1 행에서 행동이 알려짐) 기존 loss 와의 등가는 근사.
