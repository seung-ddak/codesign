# co-design 연구 전체 기록

## 1. 연구 질문 및 목표
- 4족 보행 로봇의 기본 동작에서 수집한 다중 형상 dynamics 데이터만으로, 학습하지 않은 복합 task를 수행하는 최적의 로봇 형상과 행동 trajectory를 공동 생성할 수 있는가?
- 현재 진행하려는 것은 연구 기여·신규성을 논의하는 것이 아니라, 계획한 파이프라인이 실제 작동하여 유효한 E와 A를 생성하는지 검증하는 것이다.
- 사용자가 별도로 요구하지 않으면 연구 기여, 차별성, 신규성은 논의하지 않는다.
- Co-design은 E를 먼저 생성하고 A를 후속 탐색하는 것이 아니다. TT가 노이즈에서 E, S, A를 함께 denoising하고, 미래 상태 S와 행동 A를 고려하여 task reward가 높은 E를 찾는다.
- E는 morphology, S는 상태 trajectory, A는 행동 trajectory이다.
- 최종 생성물은 (E*, S*, A*)이며, 별도의 신경망 policy 가중치를 만드는 것은 현재 목표가 아니다.
- task는 사용자가 직접 정의하고, reward와 penalty를 설계한다. TT는 task를 수행하기 위한 로봇 형상과 행동을 공동 생성한다.
- 처음 성공 기준은 생성된 E와 A가 실제 MJX에서 새로운 복합 task를 유효하게 수행하는 것이다. 처음부터 기준 형상보다 더 높은 reward를 얻어야만 성공으로 보지는 않는다.
- 최적화 목표는 reward가 높은 E와 A의 공동 생성이다. 전역 최적해 보장은 별도의 문제다.

## 2. 전체 파이프라인
1. primitive motion datasets의 데이터를 randomization하여 retargeting한다.
   - A1 URDF와 받은 motion dataset을 확인한다.
   - A1과 동일한 다리별 무릎 굽힘 방향의 TT topology 하나로 진행한다.
   - 속도(time scaling), 이동 폭, heading을 무작위화하고, 시간축을 변경하면 속도를 다시 계산한다.
   - 각 연속 morphology의 기구학에 맞게 IK retargeting한다.
2. retargeting한 primitive motion을 추종하는 PPO expert를 학습한다.
   - 잘 작동하는 모션만 사용한다.
   - primitive마다 별도 expert를 만든다.
   - 하나의 expert가 같은 design space의 여러 연속 morphology에서 해당 모션을 추종하도록 한다.
   - 미학습 치수에서 tracking 성공 여부를 확인하고 통과한 expert를 고정한다.
3. primitive motion 사이의 transition을 별도 PPO expert로 학습한다.
   - primitive expert의 가중치와 normalization은 고정한다.
   - A expert가 실제 도달한 상태에서 transition을 시작한다.
   - transition expert가 B 진입 자세·속도·접촉 상태를 맞추면 B expert로 인계한다.
   - 물리 상태를 reference 시작 상태로 순간이동시키지 않는다.
4. primitive와 transition expert를 여러 morphology에서 MJX로 실행한다.
   - 실제 물리 state와 실제 적용된 action의 결과를 수집한다.
   - E, s_t, a_t, s_(t+1), dt, terrain, done 등을 기록한다.
5. 여러 morphology의 rollout을 통합하여 dataset을 생성한다.
   - morphology–state–action–next-state 데이터셋으로 저장한다.
   - 연속된 순서, episode ID, expert ID, reference ID, random seed 등을 기록한다.
   - 임의의 다른 rollout을 연결해서 불연속한 state transition을 만들지 않는다.
6. TT joint dynamics를 학습한다.
   - 생성 목표는 p(E, S, A given s0, terrain)이다.
   - TT에서 E, S, A를 공동 생성할 수 있는 구조로 학습한다.
   - E와 A를 고정한 상태에서 미래 S를 예측하는 dynamics 검증도 진행한다.
   - one-step, H-step, 접촉 및 transition 구간을 별도로 평가한다.
7. 학습하지 않은 복합 task의 reward guidance를 진행한다.
   - 초기 노이즈에서 E, S, A를 함께 denoising한다.
   - E는 전체 복합 task 동안 고정한다.
   - 모든 denoising step이 끝난 후 E와 A를 MJX에서 open-loop로 실행한다.
   - TT 예측 S와 MJX 실제 S를 비교하고, 실제 성공 여부와 reward를 확인한다.

## 3. primitive motion datasets
### 3-1. 보유 원본 데이터
- 사용자가 제공한 raw(1).zip은 raw 폴더 내 CSV 29개로 구성되어 있다.
- CSV 데이터 행 합계는 6,779개이며 각 파일은 49개 열이다. 원본 파일의 시간열을 기준으로 프레임 간격과 모션 길이를 검토할 수 있다.
- CSV에는 time, base quaternion, IMU 정보, 관절각, 관절속도, force 센서 등이 있다.
- 이 데이터는 원본 reference이므로 세계좌표계의 실제 MJX 다음 상태, E, PPO가 출력한 action을 완전히 갖춘 TT dynamics dataset과 다르다.
- ZIP에는 trotForwardFast, trotCWSlow, walkForwardSlow, walkCWSlow, trotLeftFast/Slow, trotRightFast/Slow, walkLeftFast/Slow, walkRightFast/Slow, walkBackwardFast/Slow, squat, standUp, sitDown, tiltRoll, tiltPitch, rollBack 등의 CSV가 있다.
- howAreYou의 정확한 원본 파일 표기는 howAreYou_raw_state.csv이다.
- 원본 CSV 길이와 PPO expert를 MJX에서 실행하여 얻는 연속 rollout 길이는 서로 다르다.

### 3-2. 이번에 사용할 motion 범위
- tiltYaw와 tiltYaw 파생 클립은 사용하지 않는다.
- backFlip, howAreYou, rollBack도 사용하지 않는다.
- 현재 첫 파이프라인 검증은 점프·계단 오르기·내리기와 이들을 포함한 전환을 제외한다. 파이프라인이 연결된 후 추가한다.
- 나머지는 실제 retargeting 및 tracking 결과를 확인한 뒤 사용한다.
- 레퍼런스가 잘 작동하지 않으면 실패 원인을 기록하고 제외한다.
- 기존에 이어붙인 전환 클립과 primitive는 구분한다.
- reference randomization에서 contact timing과 지형 일치가 유지되는지 검사한다.
- 기준 형상의 관절각을 다른 길이의 로봇에 그대로 복사하지 않고 해당 형상에서 IK retargeting한다.

### 3-3. Reference 검증 항목
- 원본 A1 URDF와 reference의 관절 순서, axis, joint limit, 초기 자세를 비교한다.
- A1과 TT morphology에서 무릎이 실제로 같은 방향으로 접히는지 FK 및 시각화로 확인한다.
- IK residual, 관절 범위, 발 미끄러짐, 지면/몸체 관통, 접촉 타이밍을 검사한다.
- MJX replay로 retarget 결과를 확인한다.
- 원본 motion ID, randomization 파라미터, 형상 E, seed를 기록한다.

## 4. 기존 실험 기록과 새 파이프라인 구분
- 기존 실험은 TT 4형상 × 30모션으로 기록되어 있다. 당시 A1 원본 25, 점프·계단 2, 이어붙인 전환 3, tiltYaw 제외 구성이었다.
- 기존 방식은 형상당 정책 하나를 사용하는 형태였으며, 이번 계획의 모션별 개별 primitive expert와 다르다.
- 기존 tt_jumpBoost 기록은 112/120 통과, 넘어짐 0, jumpInPlace 종료 위치 1/4 통과이다.
- 기존 흐름은 A1 reference → TT IK retargeting → MJX PPO full-body tracking이다.
- 기존 transition 실험은 DeepMimic skill selector 방식으로 일부 episode에서 A 종료 프레임과 B 시작 프레임을 전환했다.
- B 첫 프레임 0.5초 유지 및 로봇 XY·yaw 정렬 등을 시도했고 eval_switch.py로 평가했다.
- 이 기존 성능을 새로 계획한 단일 topology·모션별 expert·별도 transition expert의 검증 완료로 간주하지 않는다.
- 기존 checkpoint, reference, 영상, 로그와 실험 결과는 유지한다.

## 5. Morphology E
### 5-1. Design space
- A1과 다리별 무릎 굽힘 방향이 동일한 TT topology 하나를 고정한다.
- FK 확인 결과 선택한 topology는 `00000011`이다. 연속 치수 중간값에서 A1/TT 질량은 12.45/13.26 kg, 다리 길이는 0.400/0.375 m이며, 새 학습·평가의 12관절 토크 한계는 A1과 같은 ±33.5 N·m로 설정한다.
- 이산 구조·관절 수·관절 순서는 변경하지 않는다.
- 같은 topology 내부에서 링크 길이, 몸체 치수 등의 연속 파라미터를 샘플링한다.
- 초기부터 여러 morphology를 사용하여 primitive·transition expert를 학습한다.
- 한 형상에서 동작한 expert가 다른 치수에서도 동작한다고 가정하지 않고 검증한다.

### 5-2. 물리적 일관성
- 앞으로 생성한 로봇 형상의 실제 제작도 고려한다.
- 허벅지나 종아리 길이가 바뀌면 질량·관성·충돌 형상도 함께 변하도록 한다.
- 실제 링크의 재료·단면·조립 구조에 맞는 일관된 물성 계산이 필요하다.
- 기존 design space의 기본 유효 범위는 유지한다.
- 추가 제약을 처음부터 임의로 강하게 넣지는 않는다.
- 생성 후 극단적인 E가 반복되면 denoising 중 제한 또는 추가 penalty를 검토한다.
- 유효하지 않은 치수, 관절 간섭, 지면 관통 등의 기본 물리 문제는 처음부터 검사한다.
- E는 하나의 trajectory 전체에서 고정한다.
- E가 바뀌면 초기 자세가 달라질 수 있으므로 morphology별 초기 state 계산과 안정화 기능이 코드에 있는지 Q16 검증 항목으로 둔다.

## 6. PPO primitive 및 transition expert
### 6-1. Primitive expert
- primitive마다 별도 PPO motion tracking expert를 학습한다.
- observation에는 현재 로봇 관측, morphology E와 reference phase/target 등이 포함될 수 있다.
- 연속 morphology의 다른 치수마다 retarget reference를 추종하도록 학습한다.
- reference에서 동작이 가능하더라도 PPO가 tracking할 수 있는지는 별도로 확인한다.
- 각 morphology에서 tracking error, 낙상, 접촉 위반, 완주 여부를 기록한다.
- 통과한 primitive expert의 가중치와 observation normalization을 고정한다.

### 6-2. Transition expert
- primitive A의 끝 상태에서 B로 넘어갈 때 물리적 상태가 자연스럽게 이어지도록 별도의 PPO expert를 학습한다.
- 방향이 있는 A→B와 B→A는 다른 전환으로 취급한다.
- 초기에는 필요한 일부 전환 쌍만 학습한다.
- source/target motion ID, E, 현재 상태, B 진입 조건을 활용한다.
- 전환은 실제 A expert가 MJX에서 만든 상태에서 시작한다.
- transition expert가 정해진 시간 안에 B 진입 자세·속도·접촉을 맞추면 고정 B expert로 넘긴다.
- B로 인계한 이후 일정 시간 tracking이 유지되는지도 검사한다.
- 전환을 위해 로봇의 실제 위치나 관절 state를 강제로 수정하지 않는다.
- 전환 실패 시 해당 E에서 A와 B 각각이 가능한지 먼저 확인한다.

### 6-3. PPO와 dynamics 데이터의 차이
- Retargeting은 주로 기구학 계산을 이용한다.
- MJX에서 진행하는 PPO motion tracking은 중력·접촉·관성 등 dynamics의 영향을 받는다.
- 그래도 PPO checkpoint 자체가 TT 학습에 필요한 명시적인 E, S, A, next-state 데이터는 아니다.
- PPO expert를 MJX에서 실제 실행하여 state transition을 저장해야 한다.
- 원본 reference 관절각과 PPO가 출력한 목표 관절각은 구분한다.

## 7. Action A와 저수준 제어
- A는 12개 관절의 목표 관절각이다.
- PPO action이 정규화 값이나 기본 자세에 대한 보정량인 경우 목표 관절각으로 바꾸는 scale 및 offset을 기록한다.
- PD controller는 목표 관절각과 현재 관절각·속도를 이용해 관절 토크를 계산한다.
- PPO rollout과 TT 생성 후 MJX 실행에는 동일한 action 변환 규칙, PD gain, torque limit, actuator 순서, 제어 주기를 사용한다.
- 실험에 실제로 기록되는 action이 원래 정책 출력인지 최종 목표 관절각인지 코드에서 다시 확인한다.
- 저수준 제어는 관절 명령을 모터 토크로 바꾸어 실행하는 부분이다.
- TT는 복합 task를 달성하기 위한 전체 관절 행동 trajectory를 생성한다.

## 8. State S 및 데이터 규약
### 8-1. State 구성
- TT의 S는 PPO observation이 아니라 실제 MJX 물리 state를 바탕으로 구성한다.
- 후보 기본 구성은 root position 3, root quaternion 4, root linear velocity 3, root angular velocity 3, joint position 12, joint velocity 12로 총 37개 값이다.
- MuJoCo에서 floating-base 12 DOF 모델이라면 qpos 19, qvel 18의 구조일 수 있다. 실제 모델에 맞는지 확인한다.
- 발 4개의 contact state를 보조 정보로 추가하면 41개 값이 된다. contact를 별도 생성 변수로 둘지는 아직 결정하지 않는다.
- 필요한 actuator 내부 state가 존재하는지도 확인한다.
- Quaternion 순서·단위·월드/몸체 좌표계·joint order는 수집과 TT 입력에서 통일한다.
- s0와 E가 함께 물리적으로 유효한 초기 상태를 나타내는지 확인한다.

### 8-2. Rollout 저장
- 매 제어 step에 E, s_t, a_t, s_(t+1), dt, terrain, done을 기록한다.
- episode ID, expert ID, 모션 ID, reference phase, seed, terminal/truncation도 저장한다.
- s_(t+1)은 action을 실제 적용한 MJX 결과로 저장한다.
- reset 이후 초기 state를 직전 action의 next-state로 기록하지 않는다.
- 실패 rollout도 정상적인 물리 전이 구간은 실패 표시와 함께 남긴다.
- 서로 다른 episode나 서로 다른 reference의 데이터를 그냥 이어붙여 가상의 전이를 만들지 않는다.
- 형상 seed와 reference 원본 ID를 기준으로 split하고, 인접 window와 randomization 파생본이 train/test에 동시에 들어가지 않도록 한다.
- 초기에는 마찰과 비형상 dynamics 조건을 통제하고, 변화를 추가하면 TT 조건과 데이터에 기록한다.

### 8-3. Expert 데이터의 범위 문제
- PPO expert가 생성한 데이터는 모든 가능한 dynamics가 아니라 expert가 방문한 state/action의 dynamics다.
- 기본 모션과 전환을 수집해도 데이터에 없는 빠른 회전이나 복합 행동을 항상 정확하게 예측한다고 보장할 수 없다.
- 우리 목표는 A처럼 primitive를 단순히 이어붙이는 것이 아니라 B처럼 reference에 없는 행동도 만들어보는 것이다.
- 예를 들어 전진·CW·CCW·측면 보행 데이터를 바탕으로 전진하면서 빠르게 방향을 바꾸는 새로운 action을 생성할 수 있는지 검증한다.
- 처음에는 정상 expert rollout만으로 TT를 학습하고 결과를 확인한다.
- 새로운 action 예측이 부정확하면 action perturbation, 복구 행동, 실패 구간 등으로 데이터 다양성을 확대하는 것을 검토한다.
- 그 추가 수집은 지금 파이프라인에 자동으로 넣지 않는다.

## 9. 시간, Timestep, Horizon
- Step은 어떤 계산 또는 과정이 한 번 진행되는 것을 의미한다. 시간 단위는 아니다.
- Physics step은 MJX에서 물리 상태를 한 번 적분하는 간격이다.
- Control step은 PPO나 제어기가 목표 관절각을 갱신하는 간격이다.
- Trajectory step은 TT가 생성하는 S와 A의 시간 해상도다.
- Diffusion denoising step은 노이즈를 줄이는 생성 반복으로 trajectory 시간 step과 다르다.
- 모든 복합 task의 시간 제한은 10초로 결정했다.
- 0.02초 간격, 50Hz는 후보이고 실제 구현에서 확정한 수치가 아니다.
- 이 가정이면 action 500개와 s0를 포함한 state 501개를 생성한다.
- Physics 500Hz, 제어 50Hz 같은 수치는 설명용 예시이며 현재 코드 설정이라고 단정하지 않는다.
- 짧은 간격을 고려하는 이유는 발 접촉 변화와 몸체 기울기 등을 충분히 표현하기 위해서다.
- TT의 한 번의 공동 생성에서 전체 10초 task를 고려하는 것이 목표다.
- 짧은 primitive의 원본 reference 길이와 실제 MJX rollout 길이는 다르다.
- 주기적인 보행 reference는 재생 경계가 연속적이라는 전제하에 expert가 더 오래 실행할 수 있다.
- 비주기적 동작을 무작정 반복하거나 별개 CSV를 이어붙여 state를 불연속하게 만들지 않는다.
- 현재 확보한 실제 rollout 길이, 10초 연속 데이터, TT의 처리 가능한 horizon은 Q11 코드 검증으로 보류했다.
- 짧은 구간의 dynamics 예측이 정확해도 10초 joint generation의 정확성이 자동 보장되지는 않는다.

## 10. TT Joint Dynamics 학습
- 목표는 p(E, S, A given s0, terrain) joint diffusion model을 학습하는 것이다.
- E는 시간 구간 내 고정이고 S, A는 시간순 데이터다.
- s0·terrain 등 알려진 조건은 mask로 보존하고 E·S·A에 diffusion을 적용한다.
- dynamics 평가에서는 E와 A를 고정하고 S를 생성하거나 복원하는 조건을 사용한다.
- task co-design에서는 E와 S와 A 모두 생성하는 조건 mask가 필요하다.
- train split에서 normalization 통계를 계산한다.
- 작은 rollout에 대한 overfit은 데이터 변환 및 학습 구현 점검용이다.
- 그다음 held-out morphology의 one-step, H-step 예측 오차를 확인한다.
- 발 접촉 변경, 착지, 전환, 방향 변경 구간은 별도 평가한다.
- 동일 E, s0, A를 MJX에 넣고 실제 S와 TT S를 비교한다.
- TT loss가 낮다고 실제 dynamics를 정확하게 재현했다고 단정하지 않는다.
- 생성과정에서 E·S·A를 같이 업데이트해야 하고, E를 먼저 고정한 후 A만 최적화하는 방식으로 바꾸지 않는다.

## 11. 복합 Task 생성 방식
### 11-1. 확정된 목표
- 첫 검증 task는 지그재그 장애물 회피 또는 슬라럼이다.
- 모든 복합 task의 제한 시간은 10초다.
- 여러 기둥을 올바른 순서와 지정된 좌우 방향으로 통과하고 목표점까지 도달하도록 만든다.
- 기둥 위치·반경·높이·간격은 실제 로봇 크기를 보고 결정한다.
- 단순히 최종 골까지 접근시키면 기둥을 우회하거나 순서를 건너뛸 수 있으므로 순서가 있는 통과 구간을 정의한다.
- 한 기둥의 특정 방향으로 지나갔는지 판단할 수 있는 통과면이나 목표 영역을 사용한다.
- 원하는 경로의 관절 trajectory를 reference로 만들어 제공하지 않는다.
- 기존 reference에 없는 몸체 비틀기·측면 이동·짧은 점프 동작이라도 실제 MJX에서 제약을 만족하면 허용한다.
- 최종 task reward에는 primitive motion tracking reward를 넣지 않는다.

### 11-2. Reward/Penalty
- 순서대로 통과한 gate 보상, 다음 gate로 이동한 진행 보상, 최종 완주 보상, 빠른 완주에 대한 보상을 사용한다.
- 기둥 충돌, 안전거리 위반, 낙상, 관절 한계, 토크 한계, 과도한 에너지 등을 penalty 후보로 둔다.
- 10초 안에 완주했는지뿐 아니라 얼마나 빨리 완주했는지도 보상에서 다룰 수 있다.
- 정지하거나 특정 구간만 반복해서 reward를 얻지 못하도록 해야 한다.
- 실제 MJX 성공 판정에는 순서·방향·충돌·낙상 등의 이산 조건을 사용한다.
- TT guidance는 gradient가 필요하므로 통과 영역과 거리, 안전 여유를 연속적인 미분 가능 함수로 근사할 수 있다.
- robot root 중심 하나만으로 충돌을 검사하지 않는다.
- E에 따라 달라지는 몸체·다리의 전체 충돌 형상과 기둥의 최소 거리로 penalty를 계산한다.
- task reward는 원하는 결과를 지정하고, 로봇의 모든 관절 움직임을 미리 지정하지 않는다.

### 11-3. Terrain과 장애물의 역할
- 원칙적으로 TT에 terrain 조건을 포함하는 구조를 생각한다.
- 하지만 처음 검증에서는 primitive/transition expert와 rollout 수집 환경에 슬라럼 기둥을 넣지 않는다.
- 최초 슬라럼 기둥은 최종 task의 reward 함수와 최종 MJX 환경에서 등장한다.
- 기둥 위치·크기는 task/reward 계산기가 알고 있어야 한다.
- TT가 기둥의 접촉 dynamics를 학습하지 않았더라도 기둥과의 거리를 이용한 penalty로 비접촉 경로 생성을 유도할 수 있는지 검증한다.
- 기둥을 접촉하지 않는 한 동일한 바닥 조건에서 평지 주행 dynamics를 활용할 수 있다.
- 기둥에 부딪힌 후의 상태 변화는 학습하지 않았으므로 정확히 예측한다고 가정하지 않는다.
- 기둥 조건을 TT 입력에 새로 넣는다고 기둥 물리를 자동으로 이해하는 것은 아니다.
- 계단처럼 장애물과 접촉해야 하는 과제로 확장하면 학습 지형과 접촉 데이터 추가가 필요할 수 있다.
- Terrain 조건 자체를 어떻게 표현하는지는 향후 코드와 데이터에서 검증한다.

### 11-4. Joint Denoising 및 MJX 최종 실행
1. 사용자가 새로운 task와 reward, penalty, 장애물 조건, 초기 상태를 정의한다.
2. 초기 노이즈에서 E, S, A를 함께 생성한다.
3. 매 denoising 단계에서 모델의 예측된 E, S, A와 task reward를 이용해 guidance한다.
4. 모든 denoising step을 마친 뒤 최종 E*, S*, A* 후보를 얻는다.
5. E와 초기 state의 물리적 유효성을 확인한다.
6. E를 고정하고 A를 MJX에서 open-loop 실행한다.
7. MJX의 실제 S, task 성공 여부와 실제 reward를 계산한다.
8. TT의 예측 S와 실제 S를 비교한다.
- 매 denoising step마다 최종 검증용 MJX를 실행하는 것이 아니다.
- Open-loop는 실행 중 실제 state를 보고 A를 다시 생성하지 않는 것이다.
- Closed-loop는 실제 상태를 관측해 남은 A를 다시 계획하는 것이며 현재 첫 검증에서는 사용하지 않는다.
- MJX가 실패했다고 원본 reference, PPO, TT를 모두 처음부터 재학습하는 것은 아니다.
- 초기에는 MJX 결과를 TT 학습에 다시 넣지 않고 평가·기록·진단에만 사용한다.
- 필요하다면 기존 학습된 TT에서 새로운 초기 노이즈로 다른 후보를 생성해 볼 수 있다.
- 새로운 후보 여러 개의 성능 비교나 MJX 결과를 이용한 재학습은 별도 후속 실험으로 구분한다.

## 12. 실패 원인 확인
### 12-1. TT 예측과 MJX 실제 실행 구분
- 생성한 E와 A를 MJX에서 실행했는데 넘어졌다면 바로 reference가 나쁘거나 reward가 잘못되었다고 단정하지 않는다.
- TT 예측 S에서도 넘어지는지, 예측은 성공인데 MJX에서만 넘어지는지 먼저 확인한다.
- TT의 S도 넘어지면 낙상 penalty, reward 항목의 상대적인 크기, guidance 강도, 생성 품질을 확인한다.
- TT S는 정상인데 MJX만 넘어지면 E·A의 dynamics 일관성, action 변환, PD 설정, 학습 분포와 다른 action을 확인한다.
- 실패 원인은 낙상 시점보다 이전에 발생할 수 있으므로 예측과 실제가 처음 크게 달라진 시점을 찾는다.

### 12-2. 우선 확인 순서
1. E의 길이·질량·관성·충돌 형상·관절 범위와 초기 상태를 확인한다.
2. TT 예측 S에서 통과·낙상·충돌 조건이 실제로 어떻게 평가되었는지 확인한다.
3. 최종 MJX에 같은 E, s0와 A가 들어갔는지 확인한다.
4. action scale, PD gain, torque limit, actuator order, physics/control timestep이 일치하는지 확인한다.
5. 몸체 위치·roll/pitch·속도·관절각·발 접촉에서 TT와 MJX 차이가 시작된 시점을 찾는다.
6. 미학습 급회전·측면 이동 등 action이 기존 expert 데이터 분포에서 벗어났는지 확인한다.
7. TT의 held-out one-step/H-step 예측 결과를 확인한다.
8. guidance 없음·약함·강함에서 예측 reward, MJX reward, 예측 오차를 비교한다.
9. 필요한 경우 reference → retarget → expert → rollout까지 역추적한다.
- guidance를 강하게 할수록 TT의 예측 reward만 증가하고 실제 MJX 성능이 나빠지면 TT 예측 오류를 이용한 허위 고보상을 의심한다.
- 최초 검증에서 문제 원인도 확인하지 않고 새로운 penalty, data collection, 재학습을 자동으로 추가하지 않는다.

## 13. 실제 코드 및 자료 확인 항목
### 13-1. 기존 파일 및 후보 경로
- Reference 및 retargeting: code/motion_tracking/pd_replay.py, 기존 TT reference·retarget 검증 코드.
- Primitive PPO: code/motion_tracking/train_rl_procedural.py, track_env.py.
- Transition PPO: track_env.py, eval_switch.py.
- Rollout: stage2/code/collect_rollouts.py, multi_design_env.py.
- TT: ../transformer-transformer/t2/data/, t2/io/, t2/model/task/, t2/eval/diffusion_guidance.py.
- 이 경로는 기존 remind에 기록된 재사용 후보다. 현재 GitHub의 최신 코드가 실제로 어떤 상태인지는 다시 확인해야 한다.
- 과거 문서에는 tracking 진입점의 continuous morphology가 중간값에 고정되고 randomization_fn = None이라고 적혀 있다. 현재 상태는 확인 전이다.

### 13-2. Q11 코드 검증으로 보류
- 기존 PPO expert가 MJX에서 한 번에 몇 초 동안 연속 실행되는지 확인한다.
- 주기적 보행 reference 반복이 state 단절 없이 처리되는지 확인한다.
- transition expert 실행 후 다음 primitive로 실제 상태가 이어지는지 확인한다.
- 10초 길이의 dynamics rollout을 이미 갖고 있는지 확인한다.
- TT 학습 입력의 sequence 길이와 모델이 처리하는 horizon이 얼마나 되는지 확인한다.
- 10초, 50Hz 수준으로 확장할 때 메모리·예측 오차가 허용되는지 확인한다.

### 13-3. Q16 코드 검증으로 보류
- morphology마다 초기 root 높이와 joint pose를 어떻게 설정하는지 확인한다.
- 형상별 발 접촉과 초기 안정화가 구현되어 있는지 확인한다.
- 생성된 E와 s0가 서로 모순되지 않는지 확인한다.
- TT 학습 초기 상태와 최종 MJX 검증 초기 상태의 규약이 같은지 확인한다.

### 13-4. 그 밖의 확인 대상
- GitHub 저장소 seung-ddak/codesign에 보유한 retarget reference, PPO checkpoint, transition checkpoint, MJX rollout, TT 관련 데이터와 실험 로그의 실제 목록을 확인한다.
- 이전에 외부 접근이 안 되어 현재 저장소 내부를 충분히 확인하지 못했다. 코드가 없거나 데이터가 없다고 단정하지 않는다.
- morphology continuous parameter와 질량·관성 갱신 코드가 이미 존재하는지 확인한다.
- PPO action 출력의 실제 변환 규칙과 PD·actuator 제어 규약을 확인한다.
- MJX의 qpos/qvel 및 발 접촉 저장 구조를 확인한다.
- 확인 이전에는 이미 구현되었을 수 있는 기능을 새로 구현하자고 단정하지 않는다.

## 14. 실험 진행과 보류 항목
- 소수 primitive·transition과 여러 morphology로 전체 파이프라인을 한 번 연결한다.
- reference 검증, expert 성공, 실제 rollout 저장, TT dynamics 검증, joint generation, 최종 MJX 순서로 진행한다.
- 이전 단계에서 심각한 문제가 확인되면 대규모 수집이나 학습으로 바로 넘어가지 않는다.
- 학습 데이터에서 완전한 슬라럼 task trajectory는 제외한다.
- 원래 primitive와 transition 데이터만으로 새로운 복합 task 행동이 나오는지 확인한다.
- 단일 topology의 연속 morphology에서 시작하고 무릎 반전 topology 비교는 나중에 한다.
- 점프·계단의 신규 학습과 관련 transition도 최초 검증 이후 추가한다.
- 처음부터 모든 방향성 transition 쌍을 학습하지 않는다.
- 처음에는 별도 정책 distillation, sim2real, 실제 제작, closed-loop 재계획을 진행하지 않는다.
- 추가 morphology 제약, action perturbation 수집, 실패 rollout 재학습, 충돌 지형 학습, dynamics consistency 제약은 최초 결과를 보고 필요할 때 검토한다.
- config, seed, 코드 버전, reference/expert 버전, reward 가중치, split, 로그, 영상, 평가 기준을 기록한다.
- 결과를 보고 기존 성공 기준을 소급해서 변경하지 않는다.

## 15. Q1~Q16 대화에서 확정 및 보류된 내용
- Q1: 복합 task 전체를 표현하는 긴 trajectory를 생성하고, 전체 task를 수행할 수 있는 E와 A를 공동 생성한다.
- Q2: 기존 모션을 단순 연결하는 것보다 기존에 없는 행동을 만드는 B 방향이 목표다. 기본 모션과 transition의 실제 dynamics를 학습하는 이유가 여기에 있다.
- Q3: 생성된 형상을 추후 실제 제작할 수 있도록 길이 변화에 따른 질량과 관성을 함께 바꾼다.
- Q4: 처음에는 유효한 E와 A가 복합 task를 성공적으로 수행하면 파이프라인 성공으로 판단한다.
- Q5: 복합 task는 모두 10초다. 발 접촉과 몸체 자세를 표현할 수 있도록 짧은 시간 간격이 필요하지만 TT step 크기는 미확정이다.
- Q5 추가: Physics step, control step, trajectory step, diffusion step을 구분한다. E, S, A는 노이즈에서 함께 생성한다.
- Q6 관련: TT의 A는 12개 관절의 목표 관절각으로 확정했다. PPO 출력값의 scale/offset은 코드 확인이 필요하다.
- Q7 관련: 정상 expert rollout만으로 새로운 행동을 먼저 검증한다. action perturbation과 복구 데이터는 필요하면 추가한다.
- Q8 관련: 최종 장애물은 기하학적 reward와 MJX 평가에 사용한다. Primitive와 transition 학습에는 새 슬라럼 기둥을 넣지 않는다.
- Q9: Action A는 목표 관절각이다.
- Q10: S는 실제 MJX 물리 state를 사용한다.
- Q11: 기존 rollout 길이와 10초 TT horizon의 구현 상태는 저장소에서 확인할 사항으로 보류한다.
- Q12: 첫 복합 task는 지그재그 장애물 회피다. 모든 복합 task 제한 시간은 10초다.
- Q13: 기존 유효 design space 내에서 먼저 생성하고, 극단적인 E가 문제가 되면 생성 중 추가 제약을 검토한다.
- Q14: 별도 dynamics consistency 제약을 처음부터 추가하지 않고 TT joint dynamics 학습 결과를 먼저 검증한다. 낙상 원인은 TT 예측 S와 MJX 실제 S를 비교하여 진단한다.
- Q15: 첫 검증에서는 MJX 결과를 평가·진단용으로만 사용하고 TT 재학습 데이터로 자동 추가하지 않는다. Open-loop는 실행 중 재계획하지 않는다는 별개 개념이다.
- Q16: morphology별 초기 자세와 안정화는 이미 코드에 있을 수 있으므로 Q11과 같이 기존 구현 확인 항목으로 보류한다.

## 16. 참고 문헌 및 용어
- DeepMimic: X. B. Peng, P. Abbeel, S. Levine, M. van de Panne, Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills, ACM Transactions on Graphics, 2018.
- Learning Agile Robotic Locomotion Skills by Imitating Animals: X. B. Peng 외, Robotics: Science and Systems, 2020.
- TT morphology는 로봇 형상이고 TT model은 Transformer Transformer를 의미한다.
- Primitive reference는 추종할 기본 동작이고, expert는 이를 물리 시뮬레이션에서 실행하는 PPO 정책이다.
- MJX rollout은 실제 action에 의해 발생한 dynamics 데이터다.
- TT joint dynamics는 E·S·A 관계를 학습하고, reward-guided denoising은 새로운 복합 task에 맞는 E·S·A를 공동 생성한다.
