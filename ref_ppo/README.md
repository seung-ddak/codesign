# ref_ppo / E_ppo (2026-11-12, 사용자 결정)

목적: 레퍼런스 모션 묶음별 PPO 트래킹 정책. 형상 5 개 (check/slide/shapes.py: E0 일반, E1 다리 짧음, E2 다리 긺, E3 몸통 짧음, E4 몸통 긺).

- 분류 (`_classification/`): CALM 10,000 개를 A1 레퍼런스 25 개 중 가장 가까운 것에 빠짐없이 배정 (구간별 최근접 → 가장 긴 시간을 차지하는 레퍼런스).
  모션이 배정된 레퍼런스: trotBackwardSlow 3533, trotRightSlow 1917, trotLeftSlow 1699, trotForwardSlow 1454, walkForwardSlow 1159, walkRightSlow 76, trotRightFast 72, trotLeftFast 71, 나머지 17 개는 0~8 (사용자 결정: 그대로 진행).
- 데이터 (`_data/`): CALM 30 Hz → 50 Hz 변환 (`a1_reference/`), 형상별 리타겟 (`references/E*/00000011/`).
- A (`ref_ppo/<레퍼런스>/`): 형상 E0~E4 를 함께 학습하는 정책 1 개. B (`E_ppo/<E>/<레퍼런스>/`): 형상 1 개 정책.
- 학습: 50 Hz / 200 Hz, 동시 환경 1,024, lr 1e-3, 256-128-64, 트래킹 보상 Peng 2020, 미끄러짐 감점 없음, 넘어지면 감점 없이 조기 종료 + step 보상 ≥ 0.
  학습량 = 5M + 모션당 1 만 step (최대 50M). 합계 A 225M, B 1125M step.
- 큐: `_tools/train_queue.py` (A 전부 → B 전부, 한 번에 하나, 끝난 것은 건너뜀). 로그 `_tools/queue.log`. 멈추면 같은 명령으로 다시 실행.
