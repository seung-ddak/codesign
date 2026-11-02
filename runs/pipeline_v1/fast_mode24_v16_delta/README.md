# v16 delta: 개입 변화량 손실 λ = 1 (remind 8-49)

- 단변수: twin + `--delta-weight 1.0 --pair-batch 16`. 매 step E 쌍 16·A 쌍 16 을 Mode 1 (E·s0·A 알려짐), 같은 시작·같은 k·같은 noise 로 넣어 x0_hat 차이 vs 실제 차이 (몸통 14 채널) MSE 를 기존 diffusion loss 에 더한다 (`tt_loss.pair_delta_mse`, `train_tt.delta_loss`).
- 안전장치: x0 공간 차이는 (1−ab)/ab 배 증폭 (5 step cosine 스케줄 k=4 에서 1e4 배) → min-SNR-γ=5 상한 (eps 등가 가중 min((1−ab)/ab, 5)). grad-norm clip 1.0 유지.
- 사전 기준 (gate.json, 기준 run = twin): (1) 틀린 A 추가 오차(속도) ≥ 0.01 이고 ≥ twin × 2 (2) E 교체 부호 일치 ≥ 0.7 (zigzag·circle) (3) 공동 생성 생존 ≥ twin (4) held-out 쌍 R²_Δ > 0 (E·A, DDIM Mode 1, `measure_delta_r2.py`) (5) 516쌍 움직임 RMSE ≤ twin × 1.1.
- 분기: A (1·2·4 통과) → λ 조정 1 회 + 공동 생성·10 s 재평가, MJX CEM 상한 실험 병행 / B (4 만 통과) → Δ 는 배우나 생성에 안 실림 → Mode 1 guidance 경로 점검 / C (4 미통과) → 표현·구조 문제 → TT 는 prior, 별도 dynamics 모델 검토.
- 주의: 학습 로그의 val_delta_*_r2 는 단일 noise 단계 x0_hat 기준이라 과대 (x_t 가 x0 를 담음). 판정은 `evaluation/delta_r2.json` (DDIM 전체) 만.
