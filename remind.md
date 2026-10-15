# co-design 연구

## 1. 연구 질문
- 4족 보행 로봇의 기본 동작에서 수집한 다중 형상 dynamics 데이터만으로, 학습하지 않은 복합 task를 수행하는 최적의 로봇 형상 E와 행동 trajectory A를 미래 상태 S까지 고려하여 공동 생성할 수 있는가?
- 현재 목표는 연구 기여 검토가 아니라 계획한 파이프라인이 실제 작동하는지 확인하는 것이다.
- Co-design은 E를 먼저 결정하고 A를 따로 만드는 것이 아니라, TT가 노이즈에서 E, S, A를 함께 생성하는 것을 의미한다.
- 최종 생성 결과는 E, S, A이다. 별도의 PPO 정책 가중치를 생성하는 것은 현재 목표가 아니다.

## 2. 전체 파이프라인
1. primitive motion datasets의 데이터를 randomization하여 retargeting한다.
   - A1과 동일한 다리별 무릎 굽힘 방향의 topology 하나만 사용한다.
   - 기존 reference를 검증하고 사용할 모션만 선택한다.
2. retargeting한 primitive motion을 추종하는 PPO expert를 학습한다.
   - 모션별 expert를 학습하고 여러 연속 morphology에서 추종하도록 한다.
3. primitive motion 사이의 transition을 별도 PPO expert로 학습한다.
   - 기존 primitive expert는 고정한다.
   - A expert → transition expert → B expert를 실제 물리 상태가 이어지도록 
     실행한다.
4. 여러 morphology에서 primitive 및 transition expert를 MJX로 실행한다.
   - 실제 dynamics rollout의 E, state, action, next-state를 수집한다.
5. 여러 morphology의 rollout을 통합하여 TT dataset을 생성한다.
   - 실제 물리 상태와 목표 관절각 action을 사용한다.
   - 연속된 trajectory와 episode 경계를 유지한다.
6. TT joint dynamics를 학습한다.
   - 목표 모델: p(E, S, A given s0, terrain).
   - one-step, H-step 및 미학습 morphology에서 dynamics 예측을 검증한다.
7. 학습하지 않은 복합 task reward로 TT joint denoising을 진행한다.
   - 초기 노이즈에서 E, S, A를 동시에 생성한다.
   - 모든 denoising이 끝난 뒤 E와 A를 MJX에서 open-loop 실행하고 실제 성공 여부를 확인한다.

## 3. 현재 최초 검증 범위
- 최초 복합 task는 기둥을 지정된 순서와 방향으로 통과하는 지그재그 장애물 회피(slalom)로 정한다.
- 앞으로 구성하는 모든 복합 task의 제한 시간은 10초로 한다.
- task는 목표 위치, 기둥 위치·반경, 통과 순서, 충돌·낙상 패널티, 완주 시간 보상 등으로 정의한다.
- 최종 reward에는 reference motion tracking 보상을 사용하지 않는다.
- 기존 reference에 없는 행동이라도 실제 MJX에서 task 조건을 만족하면 허용한다.
- 초기 실험에서는 primitive/transition 학습 환경에 슬라럼 기둥을 넣지 않는다.
- 기둥 정보는 최종 reward 계산과 MJX 검증에 사용한다. TT가 기둥 접촉 dynamics까지 학습했다고 가정하지 않는다.
- 첫 검증의 성공 기준은 생성한 E와 A가 실제 MJX에서 복합 task를 유효하게 수행하는 것이다.
- 처음에는 open-loop로 검증하고, 실행 중 재계획이나 MJX 결과를 이용한 TT 재학습은 하지 않는다.

## 4. primitive motion datasets
- A1 URDF와 원본 motion dataset으로 retargeting 가능 여부를 확인한다.
- 현재 단계에서는 tiltYaw 및 파생 클립, backFlip, howAreYou, rollBack을 제외한다.
- 점프·계단 관련 모션과 전환은 최초 파이프라인 검증 뒤에 추가한다.
- 그 외 모션도 retargeting과 PPO tracking에 실패하면 원인을 기록하고 제외한다.
- 기존 raw ZIP은 29개 CSV로 구성되어 있다. 이것은 TT용 실제 dynamics rollout이 아니라 원본 reference 데이터다.

## 5. morphology, state, action
- E는 A1과 동일한 topology 내에서 변하는 연속 치수로 구성한다.
- 링크 치수가 변하면 질량·관성·충돌 형상도 일관되게 변경한다.
- 우선 기존 design space를 사용한다. 극단적 형상이 나오면 추가 제약을 검토한다.
- 첫 topology는 `00000011`이다(FK에서 네 다리 무릎 방향이 A1과 일치). 연속 치수 중간값에서 A1/TT 질량은 12.45/13.26 kg, 다리 길이는 0.400/0.375 m다. 새 PPO 학습·평가의 12관절 토크 한계는 A1과 같은 ±33.5 N·m로 설정한다.
- S는 PPO observation이 아니라 MJX의 실제 물리 state를 사용한다.
- 기본 state 후보는 root 위치·방향·속도와 12개 관절각·관절속도이다. 발 접촉 정보는 추가로 기록한다.
- A는 12개 관절의 목표 관절각이다. PPO action scale, PD gain, 토크 한계와 제어 주기를 TT 학습·최종 MJX 실행에서 맞춘다.
- E는 하나의 trajectory 동안 고정하고 S, A는 시간에 따라 변한다.

## 6. TT 생성 및 MJX 평가
- TT는 복합 task 전체의 10초 trajectory와 E를 공동 생성한다.
- 0.02초 간격(50Hz)은 후보이며 아직 확정하지 않았다.
- Reward guidance는 전체 task의 통과·진행·완주 보상과 충돌·낙상 패널티를 사용한다.
- 충돌 패널티는 로봇 중심이 아니라 E에 따른 전체 충돌 형상과 기둥의 거리를 고려한다.
- TT guidance에는 미분 가능한 근사 reward가 필요할 수 있고, MJX 최종 성공 여부는 실제 통과·충돌·낙상으로 판정한다.
- TT가 생성한 S와 MJX로 실제 실행한 S를 비교한다.
- 실패하면 원인을 기록하고 진단한다. 기존 TT를 재학습하지 않고 다른 노이즈로 후보를 다시 생성하는 것은 가능하다.

## 7. 확인 및 검증 요소
- Q11: 실제 primitive/transition rollout 길이와 TT의 10초 학습·생성 horizon 지원 여부 확인.
- Q16: morphology별 초기 몸체 높이·관절 자세·안정화가 기존 코드에 구현되어 있는지 확인.
- 기존 checkpoint, retarget 결과, rollout dataset, TT 관련 코드와 로그의 실제 보유 상태는 GitHub 저장소에 접근한 뒤 확인.
- 목표 관절각 변환, PD gain, 제어 주기, physical state 저장 구조 확인.
- TT 생성 결과가 MJX에서 넘어지면 E·초기 상태, 예측 S, 실제 S, action 적용 규칙, dynamics 예측 오차, reward guidance 순으로 확인.
- 10초 joint generation과 학습에 없는 급회전·측면 이동의 실제 dynamics 예측은 실험 검증 대상으로 둔다.
- 기존 checkpoint·reference·실험 결과는 보존하고, 연구 계획은 임의로 변경하지 않는다.

## 8. 현재 진행 결과
- 25개 사용 대상 A1 reference를 랜덤화하고, 동일한 `00000011` topology의 연속 형상 4개에서 각각 리타겟했다. 결과와 영상은 `runs/pipeline_v1/retargeting/` 아래 형상별로 저장했다. 수치와 재현 경로는 `runs/pipeline_v1/README.md` 참조.
- `sitDown__r00`의 CPU PPO 8,192-step 실행으로 checkpoint 저장 경로를 확인했으나 평가 episode가 모두 낙상하여 expert로 사용하지 않는다.
- GPU가 다시 인식되어 `e000` 단독 `sitDown` PPO 시험 학습 및 CSV·A1 주황색/TT 파란색 비교 영상을 생성했다. 짧은 시험 결과는 낙상으로 통과하지 못했다. `e000`~`e002`의 모델과 reference를 함께 바꾸는 공유 정책 학습 경로를 추가하고 `sitDown` 시험 중이며, `e003`은 미학습 형상 평가용이다.
