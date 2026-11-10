# TT Reward Gradient vs PPO: 세 Task 비교 보고서

## 실험 방법

기존 TT의 파라미터를 동결한 상태에서 E·S·A를 생성하고, 각 후보를 10초 동안 **CPU MuJoCo**에서 오픈루프로 실행하여 점수를 계산했다. Gradient는 추론 중 미분 가능한 예측 보상을 사용하며, RL은 에피소드 단위 Gaussian residual-noise 정책을 실제 MuJoCo 보상으로 PPO 학습한다. RL은 보행 제어 PPO나 매 프레임 폐루프 제어가 아니다.

**중요:** 이 보고서의 숫자는 실제 replay 결과만 집계한다. 낙상 시 기존 코드와 동일하게 보상에서 20을 감점하며 초기 상태 무효 후보는 평균 보상에서 제외한다. 완주율의 분모는 모든 후보이다.

## 원형 트랙

**Gradient (scale=0) 실제 MuJoCo 동작 영상:** [MP4](circle/gradient_scale_0/videos/seed_20000_candidate_0.mp4)
**Gradient (scale=0) 최고 후보 E:** [0.500, 0.533, 0.474, 0.472, 0.583, 0.588, 0.631, 0.383, 0.558, 0.473, 0.521, 0.500, 0.500]; 보상 -1.632; 유효 시간 10.000초.
**Gradient (scale=1) 실제 MuJoCo 동작 영상:** [MP4](circle/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=1) 최고 후보 E:** [0.500, 0.301, 0.347, 0.664, 0.694, 0.645, 0.294, 0.303, 0.597, 0.443, 0.624, 0.500, 0.500]; 보상 -20.183; 유효 시간 6.580초.
**RL (episodic PPO) 실제 MuJoCo 동작 영상:** [MP4](circle/rl_ppo/videos/seed_20000_candidate_0.mp4)
**RL (episodic PPO) 최고 후보 E:** [0.500, 0.526, 0.479, 0.470, 0.582, 0.586, 0.626, 0.387, 0.556, 0.478, 0.518, 0.500, 0.500]; 보상 -1.591; 유효 시간 10.000초.
| 방식 | 후보 | 초기 유효 | 10초 생존 | 과제 성공 | 평균 실현 보상 | 최고 보상 | 최고 진행량 | XY 예측 오차(cm) | 관절 예측 오차(°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient (scale=0) | 2 | 2 | 2 | 0 | -1.983 | -1.632 | 0.026 | 52.046 | 6.506 |
| Gradient (scale=1) | 2 | 2 | 0 | 0 | -22.647 | -20.183 | 0.094 | 31.786 | 7.792 |
| RL (episodic PPO) | 2 | 2 | 2 | 0 | -1.977 | -1.591 | 0.021 | 52.479 | 6.489 |

- Gradient (scale=0) 종료 사유 · completed_horizon 2개
- Gradient (scale=1) 종료 사유 · fall 2개
- RL (episodic PPO) 종료 사유 · completed_horizon 2개

### 녹화된 MP4
- Gradient (scale=0) · 후보 0: [MP4](circle/gradient_scale_0/videos/seed_20000_candidate_0.mp4)
- Gradient (scale=1) · 후보 1: [MP4](circle/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
- RL (episodic PPO) · 후보 0: [MP4](circle/rl_ppo/videos/seed_20000_candidate_0.mp4)

## 지그재그 트랙

**Gradient (scale=0) 실제 MuJoCo 동작 영상:** [MP4](zigzag/gradient_scale_0/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=0) 최고 후보 E:** [0.500, 0.283, 0.335, 0.663, 0.701, 0.652, 0.288, 0.282, 0.609, 0.441, 0.635, 0.500, 0.500]; 보상 -4.100; 유효 시간 10.000초.
**Gradient (scale=1) 실제 MuJoCo 동작 영상:** [MP4](zigzag/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=1) 최고 후보 E:** [0.500, 0.339, 0.365, 0.633, 0.652, 0.601, 0.339, 0.340, 0.559, 0.470, 0.615, 0.500, 0.500]; 보상 -23.331; 유효 시간 7.100초.
**RL (episodic PPO) 실제 MuJoCo 동작 영상:** [MP4](zigzag/rl_ppo/videos/seed_20000_candidate_0.mp4)
**RL (episodic PPO) 최고 후보 E:** [0.500, 0.524, 0.482, 0.469, 0.582, 0.586, 0.625, 0.387, 0.556, 0.479, 0.516, 0.500, 0.500]; 보상 -4.504; 유효 시간 10.000초.
| 방식 | 후보 | 초기 유효 | 10초 생존 | 과제 성공 | 평균 실현 보상 | 최고 보상 | 최고 진행량 | XY 예측 오차(cm) | 관절 예측 오차(°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient (scale=0) | 2 | 2 | 2 | 0 | -4.298 | -4.100 | 0.123 | 52.046 | 6.506 |
| Gradient (scale=1) | 2 | 2 | 0 | 0 | -23.638 | -23.331 | 0.076 | 46.012 | 7.054 |
| RL (episodic PPO) | 2 | 2 | 1 | 0 | -14.186 | -4.504 | 0.031 | 43.349 | 5.888 |

- Gradient (scale=0) 종료 사유 · completed_horizon 2개
- Gradient (scale=1) 종료 사유 · fall 2개
- RL (episodic PPO) 종료 사유 · completed_horizon 1개, fall 1개

### 녹화된 MP4
- Gradient (scale=0) · 후보 1: [MP4](zigzag/gradient_scale_0/videos/seed_20000_candidate_1.mp4)
- Gradient (scale=1) · 후보 1: [MP4](zigzag/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
- RL (episodic PPO) · 후보 0: [MP4](zigzag/rl_ppo/videos/seed_20000_candidate_0.mp4)

## 최고속도 직진

**Gradient (scale=0) 실제 MuJoCo 동작 영상:** [MP4](straight/gradient_scale_0/videos/seed_20000_candidate_0.mp4)
**Gradient (scale=0) 최고 후보 E:** [0.500, 0.533, 0.474, 0.472, 0.583, 0.588, 0.631, 0.383, 0.558, 0.473, 0.521, 0.500, 0.500]; 보상 -1.787; 유효 시간 10.000초.
**Gradient (scale=1) 실제 MuJoCo 동작 영상:** [MP4](straight/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=1) 최고 후보 E:** [0.500, 0.513, 0.482, 0.605, 0.602, 0.548, 0.488, 0.499, 0.492, 0.404, 0.486, 0.500, 0.500]; 보상 -0.859; 유효 시간 10.000초.
**RL (episodic PPO) 실제 MuJoCo 동작 영상:** [MP4](straight/rl_ppo/videos/seed_20000_candidate_0.mp4)
**RL (episodic PPO) 최고 후보 E:** [0.500, 0.526, 0.479, 0.470, 0.582, 0.586, 0.626, 0.387, 0.556, 0.478, 0.518, 0.500, 0.500]; 보상 -1.760; 유효 시간 10.000초.
| 방식 | 후보 | 초기 유효 | 10초 생존 | 과제 성공 | 평균 실현 보상 | 최고 보상 | 최고 진행량 | XY 예측 오차(cm) | 관절 예측 오차(°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient (scale=0) | 2 | 2 | 2 | 0 | -2.084 | -1.787 | 0.123 | 52.046 | 6.506 |
| Gradient (scale=1) | 2 | 2 | 1 | 0 | -11.943 | -0.859 | 0.393 | 42.331 | 7.273 |
| RL (episodic PPO) | 2 | 2 | 1 | 0 | -11.762 | -1.760 | -0.009 | 42.836 | 5.863 |

- Gradient (scale=0) 종료 사유 · completed_horizon 2개
- Gradient (scale=1) 종료 사유 · fall 1개, completed_horizon 1개
- RL (episodic PPO) 종료 사유 · completed_horizon 1개, fall 1개

### 녹화된 MP4
- Gradient (scale=0) · 후보 0: [MP4](straight/gradient_scale_0/videos/seed_20000_candidate_0.mp4)
- Gradient (scale=1) · 후보 1: [MP4](straight/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
- RL (episodic PPO) · 후보 0: [MP4](straight/rl_ppo/videos/seed_20000_candidate_0.mp4)

## 판정 기준

운동학적 일관성만으로 성공을 선언하지 않는다. 10초 생존과 과제별 성공을 분리한다. legacy 행은 이전 평가기의 시간 판정을 표시한다. 후보 수가 작거나 RL 평가 Seed가 적으면 차이를 통계적으로 확정하지 않는다.

## 실험 한계

RL은 에피소드 보상을 사용한 저차원 residual-noise PPO이며, 상태별 보행 PPO나 TT 재학습이 아니다. TT 예측 상태가 물리적으로 일관되지 않을 수 있으므로 실현된 보상만 성능으로 인정한다.

## 영상 녹화 요약

확인된 MP4 9개 / 파일 없음 9개
영상은 report.html에 포함됩니다. 파일을 브라우저에서 직접 여세요.

