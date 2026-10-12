# cp-design 연구

## 1. 연구 질문
- 4족 보행 로봇의 기본 동작 만을 이용하여 보지 못한 복합 task를 수행할 수 있는 최적의 형상과 정책을 만들 수 있을까?

## 2. 전체 파이프라인
    1. primitive motion datasets의 데이터를 랜더마이즈 하여 retargeting한다.
        - motion을 randomization 
        - 16개 무릎 조합에 대해 Retargeting
    2. retrageting하여 primivie motions tracking ppo expert
        - ppo motion tracking expert를 학습
    3. primitive motions's transition도 ppo expert
        - jump, stair은 a1의 오픈소스 설계 사항으로 RL로 학습하여 모션 확보
        - expert간 transition도 학습
    4. primitive motions motion tracking expert, transition expert로 여러 morphology에서 tracking 
       rollout data 획득
        - expert를 MJX에서 실행하여 실제 dynamics 데이터 수집
        - rollout 데이터 수집    
    4. 여러 morphology rollout 통합 -> morphology–state–action–next-state dataset 생성
    5. TT joint dynamics 학습 -> p(E, S, A | s0, terrain) 생성 모델, held-out one-step·H-step 예측
    6. 보지 못한 복합 task reward guidance


### 2-1 primitive motion datasets
- a1 urdf와 받은 motion dataset을 이용해서 모션 데이터셋의 움직임이 a1 urdf가 잘 추종하는지 확인
- 오픈 소스 활용하여 a1의 기본 동작을 가지고 점프 및 계단 오르고 내리기 PPO RL로 학습하여 모션 생성
- 생성된 보행, 점프, 계단 오르기 모션 retargeting

### 2-2 retargeting하여 motion tracking ppo expert
- a1의 모션을 따라하는 다양한 
- **tracking 현황 (2026-10-12)**: TT 4 형상 × 30 모션 (A1 원본 25 + 점프·계단 2 + 이어붙인 전환 3, tiltYaw 제외), 형상당 정책 1 개. 최신 tt_jumpBoost 통과 112/120 · 넘어짐 0, jumpInPlace 끝 위치만 1/4 통과
- 방법: A1 레퍼런스 → TT IK 리타겟 → MJX PPO full-body tracking (보상 [2] ×10, RSI·조기 종료 [1], kp 30, 13.3 kg). 100M 후 점프 3 모션 샘플링 6 배로 20M 이어 학습
- 진행 중: 목표 바꾸기 전환 학습 (episode 50%, 계단·held-out 회전 보행→점프 제외). 기록 runs/TT_tracking/report.html

## 참고 문헌
1. X. B. Peng, P. Abbeel, S. Levine, M. van de Panne, "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills," ACM Trans. Graph. (SIGGRAPH) 37(4), 2018.
2. X. B. Peng, E. Coumans, T. Zhang, T.-W. Lee, J. Tan, S. Levine, "Learning Agile Robotic Locomotion Skills by Imitating Animals," Robotics: Science and Systems (RSS), 2020.
