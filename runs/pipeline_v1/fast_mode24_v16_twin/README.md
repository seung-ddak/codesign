# v16 twin: Δ-loss A/B 의 대조군 (remind 8-49)

- 설정: v12 와 같은 구조·학습 설정 (modes 1/2/4 = 20/40/40, film, 60k step, seed 0) 에 kp30 세계 데이터 (wide16_action 16 + e_swap/full 16 + action_pairs_wide16 16). `--delta-weight 0` → 쌍 손실 없음, 쌍 R²_Δ 는 모니터만 기록.
- 역할: `fast_mode24_v16_delta` 와 데이터·구조·seed·step 이 전부 같고 λ 만 0. delta 의 사전 기준 (틀린 A ≥ twin × 2, 생존 ≥ twin, RMSE ≤ twin × 1.1) 의 기준값을 제공.
- 생성: `python runs/pipeline_v1/prepare_v16.py` (conditions.json·data_audit.json·run_train.py). 학습 `run_train.py`, 평가 `../eval_v12parity/run_eval.py` (kp30, reference_run = v12 는 정보용. v12 는 PD kp200 이라 직접 비교 안 함).
