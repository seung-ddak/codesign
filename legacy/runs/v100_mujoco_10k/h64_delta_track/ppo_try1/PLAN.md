# PPO 잔차 학습 try1 (2026-11-10, 사용자 결정)
동결 TT = h64_delta_track. task 원·슬라럼·직진 각각 별도 프로세스 (CPU 4코어씩).
PPO 150 업데이트 × 8 rollout, lr 0.001, epoch 4. 최종 평가 후보 4 × seed 2, 무보정 (scale 0)·gradient (scale 1) 함께 평가.
비교: task_v1 (PPO 2 × 4). 예상 약 45 분.
판정 (사전 고정): PPO 가 무보정·gradient 대비 실제 진행량·10 초 생존·과제 성공을 개선하는가. 보상만 오르고 진행량 음수면 실패.
