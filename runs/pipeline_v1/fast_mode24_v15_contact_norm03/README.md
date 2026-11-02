# v15: v12 + 접지 채널 minmax 정규화 ±1 → ±0.3 (remind 8-43 (2))

- 단변수: `train_tt.py --contact-range 0.3` (신설 플래그, 기본 1.0 = 기존). minmax 뒤 `sd[38:42] /= contact_range` 한 줄. 손실은 기존 그대로 (v13/v14 의 그룹 가중 없음).
- 체크포인트 mu/sd 에 반영되므로 measure_given_a / evaluate_native / generate_task 평가는 코드 변경 없이 일관. 2 step 스모크로 sd[38:42] = (0.5+1e-3)/0.3 확인.
- 데이터·구조·Mode·seed·60k step = v12. 학습 `run_train.py` → 평가 `../eval_v12parity/run_eval.py` → `gate.json`, `report.html`.
- 주의: train_tt.py 수정으로 v13 `conditions.json` 의 implementation_sha256 (train_tt.py) 은 더 이상 현재 파일과 맞지 않는다 (v13 build_report.py 재실행 시 실패). 동작 변화는 플래그 생략 시 없음.
