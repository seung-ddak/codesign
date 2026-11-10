# TT Reward Gradient vs PPO: 세 Task 비교 보고서

## 실험 방법

기존 TT의 파라미터를 동결한 상태에서 E·S·A를 생성하고, 각 후보를 10초 동안 **CPU MuJoCo**에서 오픈루프로 실행하여 점수를 계산했다. Gradient는 추론 중 미분 가능한 예측 보상을 사용하며, RL은 에피소드 단위 Gaussian residual-noise 정책을 실제 MuJoCo 보상으로 PPO 학습한다. RL은 보행 제어 PPO나 매 프레임 폐루프 제어가 아니다.

**중요:** 이 보고서의 숫자는 실제 replay 결과만 집계한다. 낙상 시 기존 코드와 동일하게 보상에서 20을 감점하며 초기 상태 무효 후보는 평균 보상에서 제외한다. 완주율의 분모는 모든 후보이다.

## 원형 트랙

**Gradient (scale=0) 실제 MuJoCo 동작 영상:** [MP4](circle/gradient_scale_0/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=0) 최고 후보 E:** [0.500, 0.502, 0.456, 0.707, 0.650, 0.504, 0.506, 0.501, 0.494, 0.332, 0.711, 0.500, 0.500]; 보상 -0.310; 유효 시간 10.000초.
**Gradient (scale=1) 실제 MuJoCo 동작 영상:** [MP4](circle/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=1) 최고 후보 E:** [0.500, 0.503, 0.481, 0.589, 0.557, 0.500, 0.503, 0.502, 0.491, 0.591, 0.557, 0.500, 0.500]; 보상 -0.267; 유효 시간 10.000초.
**RL (episodic PPO) 실제 MuJoCo 동작 영상:** [MP4](circle/rl_ppo/videos/seed_20000_candidate_1.mp4)
**RL (episodic PPO) 최고 후보 E:** [0.500, 0.502, 0.457, 0.706, 0.649, 0.504, 0.506, 0.501, 0.494, 0.332, 0.712, 0.500, 0.500]; 보상 -0.307; 유효 시간 10.000초.
| 방식 | 후보 | 초기 유효 | 10초 생존 | 과제 성공 | 평균 실현 보상 | 최고 보상 | 최고 진행량 | XY 예측 오차(cm) | 관절 예측 오차(°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient (scale=0) | 2 | 2 | 2 | 0 | -0.470 | -0.310 | -0.019 | 17.133 | 3.891 |
| Gradient (scale=1) | 2 | 2 | 2 | 0 | -0.528 | -0.267 | -0.027 | 15.299 | 4.834 |
| RL (episodic PPO) | 2 | 2 | 2 | 0 | -0.477 | -0.307 | -0.019 | 16.912 | 3.895 |

- Gradient (scale=0) 종료 사유 · completed_horizon 2개
- Gradient (scale=1) 종료 사유 · completed_horizon 2개
- RL (episodic PPO) 종료 사유 · completed_horizon 2개

### 녹화된 MP4
- Gradient (scale=0) · 후보 1: [MP4](circle/gradient_scale_0/videos/seed_20000_candidate_1.mp4)
- Gradient (scale=1) · 후보 1: [MP4](circle/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
- RL (episodic PPO) · 후보 1: [MP4](circle/rl_ppo/videos/seed_20000_candidate_1.mp4)

## 지그재그 트랙

**Gradient (scale=0) 실제 MuJoCo 동작 영상:** [MP4](zigzag/gradient_scale_0/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=0) 최고 후보 E:** [0.500, 0.502, 0.456, 0.707, 0.650, 0.504, 0.506, 0.501, 0.494, 0.332, 0.711, 0.500, 0.500]; 보상 -2.205; 유효 시간 10.000초.
**Gradient (scale=1) 실제 MuJoCo 동작 영상:** [MP4](zigzag/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=1) 최고 후보 E:** [0.500, 0.503, 0.524, 0.653, 0.600, 0.503, 0.500, 0.505, 0.496, 0.547, 0.502, 0.500, 0.500]; 보상 -2.372; 유효 시간 10.000초.
**RL (episodic PPO) 실제 MuJoCo 동작 영상:** [MP4](zigzag/rl_ppo/videos/seed_20000_candidate_1.mp4)
**RL (episodic PPO) 최고 후보 E:** [0.500, 0.502, 0.457, 0.706, 0.649, 0.504, 0.506, 0.501, 0.494, 0.333, 0.711, 0.500, 0.500]; 보상 -2.200; 유효 시간 10.000초.
| 방식 | 후보 | 초기 유효 | 10초 생존 | 과제 성공 | 평균 실현 보상 | 최고 보상 | 최고 진행량 | XY 예측 오차(cm) | 관절 예측 오차(°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient (scale=0) | 2 | 2 | 2 | 0 | -2.539 | -2.205 | -0.115 | 17.133 | 3.891 |
| Gradient (scale=1) | 2 | 2 | 2 | 0 | -2.601 | -2.372 | -0.176 | 13.797 | 4.479 |
| RL (episodic PPO) | 2 | 2 | 2 | 0 | -2.534 | -2.200 | -0.113 | 17.072 | 3.887 |

- Gradient (scale=0) 종료 사유 · completed_horizon 2개
- Gradient (scale=1) 종료 사유 · completed_horizon 2개
- RL (episodic PPO) 종료 사유 · completed_horizon 2개

### 녹화된 MP4
- Gradient (scale=0) · 후보 1: [MP4](zigzag/gradient_scale_0/videos/seed_20000_candidate_1.mp4)
- Gradient (scale=1) · 후보 1: [MP4](zigzag/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
- RL (episodic PPO) · 후보 1: [MP4](zigzag/rl_ppo/videos/seed_20000_candidate_1.mp4)

## 최고속도 직진

**Gradient (scale=0) 실제 MuJoCo 동작 영상:** [MP4](straight/gradient_scale_0/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=0) 최고 후보 E:** [0.500, 0.502, 0.456, 0.707, 0.650, 0.504, 0.506, 0.501, 0.494, 0.332, 0.711, 0.500, 0.500]; 보상 -0.305; 유효 시간 10.000초.
**Gradient (scale=1) 실제 MuJoCo 동작 영상:** [MP4](straight/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=1) 최고 후보 E:** [0.500, 0.503, 0.518, 0.662, 0.616, 0.503, 0.500, 0.505, 0.496, 0.541, 0.498, 0.500, 0.500]; 보상 -0.321; 유효 시간 10.000초.
**RL (episodic PPO) 실제 MuJoCo 동작 영상:** [MP4](straight/rl_ppo/videos/seed_20000_candidate_1.mp4)
**RL (episodic PPO) 최고 후보 E:** [0.500, 0.502, 0.457, 0.706, 0.649, 0.504, 0.506, 0.501, 0.494, 0.332, 0.712, 0.500, 0.500]; 보상 -0.302; 유효 시간 10.000초.
| 방식 | 후보 | 초기 유효 | 10초 생존 | 과제 성공 | 평균 실현 보상 | 최고 보상 | 최고 진행량 | XY 예측 오차(cm) | 관절 예측 오차(°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient (scale=0) | 2 | 2 | 2 | 0 | -0.471 | -0.305 | -0.115 | 17.133 | 3.891 |
| Gradient (scale=1) | 2 | 2 | 2 | 0 | -0.654 | -0.321 | -0.180 | 14.855 | 4.403 |
| RL (episodic PPO) | 2 | 2 | 2 | 0 | -0.464 | -0.302 | -0.113 | 17.247 | 3.898 |

- Gradient (scale=0) 종료 사유 · completed_horizon 2개
- Gradient (scale=1) 종료 사유 · completed_horizon 2개
- RL (episodic PPO) 종료 사유 · completed_horizon 2개

### 녹화된 MP4
- Gradient (scale=0) · 후보 1: [MP4](straight/gradient_scale_0/videos/seed_20000_candidate_1.mp4)
- Gradient (scale=1) · 후보 1: [MP4](straight/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
- RL (episodic PPO) · 후보 1: [MP4](straight/rl_ppo/videos/seed_20000_candidate_1.mp4)

## 판정 기준

운동학적 일관성만으로 성공을 선언하지 않는다. 10초 생존과 과제별 성공을 분리한다. legacy 행은 이전 평가기의 시간 판정을 표시한다. 후보 수가 작거나 RL 평가 Seed가 적으면 차이를 통계적으로 확정하지 않는다.

## 실험 한계

RL은 에피소드 보상을 사용한 저차원 residual-noise PPO이며, 상태별 보행 PPO나 TT 재학습이 아니다. TT 예측 상태가 물리적으로 일관되지 않을 수 있으므로 실현된 보상만 성능으로 인정한다.

## 영상 녹화 요약

확인된 MP4 9개 / 파일 없음 9개
영상은 report.html에 포함됩니다. 파일을 브라우저에서 직접 여세요.

