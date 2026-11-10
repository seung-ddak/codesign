# e0000 PPO 트래킹 데이터로 TT 학습 → task (2026-11-10, 사용자 결정)
데이터: runs/calm_dynamics_1000x100/rollouts/e0000/primitive_seed0.npz — e0000 형상, 리타겟 CALM 모션 100 개를 형상 전용 PPO (5.24M step) 로 실행. 50 Hz, 끝까지 73 · 넘어짐 27, 유효 전이 70,079, 추종 합격 0/100 (관절 중앙값 9.0°, 위치 0.35 m).
1. 비교 영상 (먼저): A1 원본 | TT PPO 트래킹 실제 | (TT 학습 후) TT 예측. 메쉬 렌더링 (로컬 OSMesa, env_osmesa.sh). 6 개: 추종 최상 2 · 중간 2 · 넘어짐 2.
2. TT: h64_delta_track 과 같은 설정 (H64, delta, 60k step, batch 64), 데이터만 이것. 데이터가 2.4 % 라 과적합 가능 → 검증 손실로 판단.
3. task: reward gradient 디노이징 vs RL 디노이징 (benchmark gradient/rl), task_v1 과 같은 명령, 50 Hz. 보상은 통일된 점수식 (12:3x 변경).

## task_v2: 보상 조정 (사용자 결정 2026-11-10 13:3x, TT 그대로)
- 진단: 버틴 후보에서 기울기·높이 감점 (×20) 이 진행 점수보다 큼 → 유도가 제자리 쪽으로. 실제 넘어짐은 생존 감점이 이미 잡음.
- TT_REWARD_PROFILE=v2: 진행 ×10 (1 m 당, 원형 1 rad = 1 m), 기울기·높이 ×2. 나머지 (이탈 20, 떨림 0.05, 생존 20, 폴 20) 그대로. 유도와 채점에 같은 식.
- 유도 강도 0/1/3/10, 후보 4 × seed 2, RL 업데이트 20 × rollout 4.
- 판정 (사전 고정): 생존 유지 + 진행량 증가면 유효. 진행 증가에 생존이 크게 떨어지면 기울기 감점 재조정.
- 영상: task 별 reward gradient (모든 유도 강도 중 최고) | RL, videos/task_*_gradient_vs_rl_v2.mp4

## tt_d100: 디노이징 100 단계 (사용자 결정 2026-11-10 14:3x)
- 원본 TT 설계 (hardware_gen) 와 train_tt 기본값 = 100 단계. 지금까지 TT 는 속도 때문에 5 단계.
- e0000 TT 를 --diff-steps 100 으로만 바꿔 재학습 (나머지·데이터 동일) → tt_d100/.
- 다음: 예측 지평별 정확도 (tt_horizon.py) 와 task_v2 와 같은 평가를 100 단계로 반복 (생성 비용 약 20 배 → 평가 범위는 생성 1 회 시간 측정 후 결정).
