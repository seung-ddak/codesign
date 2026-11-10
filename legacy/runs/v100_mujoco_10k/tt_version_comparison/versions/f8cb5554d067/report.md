# TT Reward Gradient vs PPO: 세 Task 비교 보고서

## 실험 방법

기존 TT의 파라미터를 동결한 상태에서 E·S·A를 생성하고, 각 후보를 10초 동안 **CPU MuJoCo**에서 오픈루프로 실행하여 점수를 계산했다. Gradient는 추론 중 미분 가능한 예측 보상을 사용하며, RL은 에피소드 단위 Gaussian residual-noise 정책을 실제 MuJoCo 보상으로 PPO 학습한다. RL은 보행 제어 PPO나 매 프레임 폐루프 제어가 아니다.

**중요:** 이 보고서의 숫자는 실제 replay 결과만 집계한다. 낙상 시 기존 코드와 동일하게 보상에서 20을 감점하며 초기 상태 무효 후보는 평균 보상에서 제외한다. 완주율의 분모는 모든 후보이다.

## 원형 트랙

**Gradient (scale=0) 실제 MuJoCo 동작 영상:** [MP4](circle/gradient_scale_0/videos/seed_20000_candidate_0.mp4)
**Gradient (scale=0) 최고 후보 E:** [0.500, 0.571, 0.376, 0.499, 0.551, 0.611, 0.478, 0.411, 0.471, 0.499, 0.614, 0.500, 0.500]; 보상 -18.543; 유효 시간 5.220초.
**Gradient (scale=1) 실제 MuJoCo 동작 영상:** [MP4](circle/gradient_scale_1/videos/seed_20000_candidate_0.mp4)
**Gradient (scale=1) 최고 후보 E:** [0.500, 0.531, 0.207, 0.443, 0.540, 0.614, 0.148, 0.498, 0.061, 0.878, 0.846, 0.500, 0.500]; 보상 -23.408; 유효 시간 0.840초.
**RL (episodic PPO) 실제 MuJoCo 동작 영상:** [MP4](circle/rl_ppo/videos/seed_20000_candidate_0.mp4)
**RL (episodic PPO) 최고 후보 E:** [0.500, 0.576, 0.373, 0.504, 0.555, 0.611, 0.495, 0.404, 0.483, 0.487, 0.612, 0.500, 0.500]; 보상 -18.469; 유효 시간 5.040초.
| 방식 | 후보 | 초기 유효 | 10초 생존 | 과제 성공 | 평균 실현 보상 | 최고 보상 | 최고 진행량 | XY 예측 오차(cm) | 관절 예측 오차(°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient (scale=0) | 2 | 2 | 0 | 0 | -21.243 | -18.543 | 0.329 | 65.158 | 8.665 |
| Gradient (scale=1) | 2 | 2 | 0 | 0 | -23.587 | -23.408 | 0.060 | 36.255 | 9.437 |
| RL (episodic PPO) | 2 | 2 | 0 | 0 | -21.572 | -18.469 | 0.342 | 65.113 | 8.292 |

- Gradient (scale=0) 종료 사유 · fall 2개
- Gradient (scale=1) 종료 사유 · fall 2개
- RL (episodic PPO) 종료 사유 · fall 2개

### 녹화된 MP4
- Gradient (scale=0) · 후보 0: [MP4](circle/gradient_scale_0/videos/seed_20000_candidate_0.mp4)
- Gradient (scale=1) · 후보 0: [MP4](circle/gradient_scale_1/videos/seed_20000_candidate_0.mp4)
- RL (episodic PPO) · 후보 0: [MP4](circle/rl_ppo/videos/seed_20000_candidate_0.mp4)

## 지그재그 트랙

**Gradient (scale=0) 실제 MuJoCo 동작 영상:** [MP4](zigzag/gradient_scale_0/videos/seed_20000_candidate_0.mp4)
**Gradient (scale=0) 최고 후보 E:** [0.500, 0.571, 0.376, 0.499, 0.551, 0.611, 0.478, 0.411, 0.471, 0.499, 0.614, 0.500, 0.500]; 보상 -24.105; 유효 시간 5.220초.
**Gradient (scale=1) 실제 MuJoCo 동작 영상:** [MP4](zigzag/gradient_scale_1/videos/seed_20000_candidate_0.mp4)
**Gradient (scale=1) 최고 후보 E:** [0.500, 0.306, 0.578, 0.425, 0.672, 0.566, 0.650, 0.376, 0.531, 0.642, 0.462, 0.500, 0.500]; 보상 -23.121; 유효 시간 0.880초.
**RL (episodic PPO) 실제 MuJoCo 동작 영상:** [MP4](zigzag/rl_ppo/videos/seed_20000_candidate_0.mp4)
**RL (episodic PPO) 최고 후보 E:** [0.500, 0.576, 0.373, 0.504, 0.555, 0.611, 0.494, 0.404, 0.483, 0.487, 0.613, 0.500, 0.500]; 보상 -23.971; 유효 시간 4.740초.
| 방식 | 후보 | 초기 유효 | 10초 생존 | 과제 성공 | 평균 실현 보상 | 최고 보상 | 최고 진행량 | XY 예측 오차(cm) | 관절 예측 오차(°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient (scale=0) | 2 | 2 | 0 | 0 | -24.806 | -24.105 | 0.791 | 65.158 | 8.665 |
| Gradient (scale=1) | 2 | 2 | 0 | 0 | -24.183 | -23.121 | 0.348 | 25.483 | 9.730 |
| RL (episodic PPO) | 2 | 2 | 0 | 0 | -25.071 | -23.971 | 0.728 | 61.058 | 8.398 |

- Gradient (scale=0) 종료 사유 · fall 2개
- Gradient (scale=1) 종료 사유 · fall 2개
- RL (episodic PPO) 종료 사유 · fall 2개

### 녹화된 MP4
- Gradient (scale=0) · 후보 0: [MP4](zigzag/gradient_scale_0/videos/seed_20000_candidate_0.mp4)
- Gradient (scale=1) · 후보 0: [MP4](zigzag/gradient_scale_1/videos/seed_20000_candidate_0.mp4)
- RL (episodic PPO) · 후보 0: [MP4](zigzag/rl_ppo/videos/seed_20000_candidate_0.mp4)

## 최고속도 직진

**Gradient (scale=0) 실제 MuJoCo 동작 영상:** [MP4](straight/gradient_scale_0/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=0) 최고 후보 E:** [0.500, 0.433, 0.567, 0.447, 0.573, 0.607, 0.532, 0.438, 0.550, 0.527, 0.475, 0.500, 0.500]; 보상 -24.024; 유효 시간 2.120초.
**Gradient (scale=1) 실제 MuJoCo 동작 영상:** [MP4](straight/gradient_scale_1/videos/seed_20000_candidate_1.mp4)
**Gradient (scale=1) 최고 후보 E:** [0.500, 0.441, 0.592, 0.452, 0.577, 0.603, 0.549, 0.480, 0.534, 0.555, 0.475, 0.500, 0.500]; 보상 -23.310; 유효 시간 1.940초.
**RL (episodic PPO) 실제 MuJoCo 동작 영상:** [MP4](straight/rl_ppo/videos/seed_20000_candidate_1.mp4)
**RL (episodic PPO) 최고 후보 E:** [0.500, 0.430, 0.563, 0.449, 0.575, 0.607, 0.528, 0.435, 0.552, 0.526, 0.477, 0.500, 0.500]; 보상 -24.660; 유효 시간 2.280초.
| 방식 | 후보 | 초기 유효 | 10초 생존 | 과제 성공 | 평균 실현 보상 | 최고 보상 | 최고 진행량 | XY 예측 오차(cm) | 관절 예측 오차(°) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gradient (scale=0) | 2 | 2 | 0 | 0 | -25.352 | -24.024 | 0.791 | 65.158 | 8.665 |
| Gradient (scale=1) | 2 | 2 | 0 | 0 | -23.391 | -23.310 | 0.302 | 27.264 | 9.059 |
| RL (episodic PPO) | 2 | 2 | 0 | 0 | -26.308 | -24.660 | 0.650 | 66.085 | 8.374 |

- Gradient (scale=0) 종료 사유 · fall 2개
- Gradient (scale=1) 종료 사유 · fall 2개
- RL (episodic PPO) 종료 사유 · fall 2개

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

