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
3. transition은 별도 정책 없이 고정된 primitive expert를 이어 붙여 데이터로 확보한다 (2026-10-17 결정).
   - TT는 dynamics를 배우고 최종 실행은 MJX open-loop이므로 expert hand-off 정책이 필요 없다. 필요한 것은 전환 구간 데이터다.
   - DeepMimic composite 방식: 10초 sequence 안에서 무작위 시점에 무작위 모션 expert로 넘긴다.
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
- 리타겟 IK가 몸체 높이 목표를 약하게 반영하여 앉기 깊이를 압축하는 문제를 수정하고 4개 형상 × 25개 reference를 재생성했다. 기존 PPO checkpoint와 평가는 이전 reference를 사용했으므로 새 reference 기준 expert 학습은 다시 해야 한다. PPO 환경의 RSI(무작위 클립·프레임 시작, 매 종료 시 새 reset)는 이미 활성화되어 있다.
- 새 리타겟의 깊은 앉기에서는 비발 형상(종아리 모터·허벅지)이 MuJoCo 바닥을 관통한다. `runs/pipeline_v1/retargeting/collision_audit.csv`에서 100개 중 15개 클립이 5 mm 초과 관통한다. 발 고정 해제 시험은 관통을 해결하지 못했다. 관통 클립은 충돌 제약을 포함해 다시 리타겟하기 전까지 expert 학습에서 사용하지 않는다.
- `walkBackwardSlow__r00` 공유 PPO expert는 e000~e002 세 학습 형상에서 전체 모션 추종 기준을 통과했고 미학습 e003은 불합격이다. 현재 후보 목록(`runs/pipeline_v1/experts/selection.csv`)에는 fast/slow/walk/trot 구분 없이 네 TT 형상 모두에서 비발 형상 바닥 관통 기록값이 0 mm인 19개 모션을 선정했다. 이 기준만으로 동역학적 추종 가능성이 확인되지는 않는다. 진행 중이던 `walkForwardSlow` 및 `trotBackwardFast` 학습은 중단했고 이번에는 목록만 기록했다.
- 선정된 19개 모션 expert 학습용으로 평지 전체 형상 충돌을 활성화하고 MJX 계단 충돌은 발만 유지했다. 세 형상 GPU reset/step과 `squat` 12,288-step 소규모 학습·e000 CSV/영상 평가 경로를 확인했다. 장비와 설정은 `runs/pipeline_v1/experts/HARDWARE.md`에 기록하고 `train_selected_experts.sh`로 모션별 순차 학습을 실행한다.

### 8-1. 선정 모션 expert (Issue #2, 2026-10-15)
- 19개 모션 통과: e000 19/19, e001 16/19, e002 14/19, 미학습 e003 0/19. 체크포인트 19개 모두 GitHub `runs/pipeline_v1/experts/`.
- 학습 형상 3개로는 미학습 형상에 일반화되지 않는다 → 다형상 수집 시 학습 형상 수가 핵심 변수.
- 11/19 run이 학습 말기 KL > 1 (최대 1.2e4, entropy → 0). e001·e002 불합격 모션과 겹친다. 원인 미확인.

### 8-2. e000 rollout 수집 (2026-10-17, 계획·조건 `runs/pipeline_v1/ROLLOUT_PLAN.md`)
- 결론: 전환 정책 없이 19개 expert를 이어 붙인 10초 sequence로 수집한다. 시험(모션 4개, 8 ep) 7/8 완주, 전환 4.9회/ep, 전환 직후 관절 오차 2~6°.
- 구성: 무작위 모션·프레임 시작 → 1~3 s마다 무작위 다른 모션 (2발 이상 접지 프레임에서만, 진입 = 관절각·접지 최근접 프레임, B reference를 로봇 xy·yaw에 정렬, 클립 끝이면 즉시 전환) → 10 s 또는 낙상까지.
- 환경: e000, 50 Hz 제어, kick·초기 교란 없음, obs noise 학습값, 결정적 정책, 낙상만 종료. 낙상 구간도 `fell`·`valid`로 표시해 저장.
- 저장: `rollouts/e000/seq_seed<N>.npz` (qpos·qvel·contact·action·target_q·clip·ref_frame·switched·valid, E) + `_pairs.csv` (pair별 1 s 내 낙상).

### 8-3. 로컬 e000 수집 판정 (2026-10-17)
- 수집기 수정: 마지막 전환 집계, 제한된 PD 목표 저장, 수치 오류 차단, 날짜 기반 run 선택, 기존 출력 덮어쓰기 방지. 회귀 검사·NPZ 검증 통과.
- 19 expert smoke 8 ep 중 5 낙상. seed 0은 256 ep 중 106 낙상(41.4%), 150 완주; 유효 104,465 step, 전환 1,208회. `runs/pipeline_v1/rollouts/e000/seq_seed0.npz` 보존.
- ROLLOUT_PLAN 분기 C에 따라 seed 1~3 중단. 전환 후 1 s 내 낙상 98, 그 외 8; pair 제외 없음. 일반화 검증이나 TT 학습 완료를 뜻하지 않는다.

### 8-4. 진입 프레임 매칭 비교 (e000, seed 0, 256 ep, 2026-10-17)
- 결론: 매칭 방식은 낙상 원인이 아니다. 전환 후 1 s 내 낙상 = 기존 8.1% / 속도 반영 표 6.3% / 실제 state 매칭 7.0% (seed 1 개, 차이 유의하지 않음).
- 실제 state 매칭에서도 진입 속도차 median 0.19 m/s 불변 → B 클립에 맞는 속도 프레임이 없다. 낙상 = 모션 간 속도 간극 + expert 견고성 (walkForwardFast 도착 ~42% 고정).
- 결정 (사용자): 기존 데이터 그대로 사용 (가). 비교 데이터 `rollouts_archive/velmatch/`, `rollouts_archive/livematch/` 보존. 옵션 `--switch-entry live`, `--match-vel-weight` (기본값 = 기존).

### 8-5. TT dynamics 학습 (npz 직접, e000~e002 rollout 1,680 ep, 2026-10-17)
- 결론: 데이터에는 몸통 dynamics 정답이 충분하다 (1-step MLP 32 step 예측 yaw 4.0°·위치 2.2 cm). TT 는 아직 못 뽑아낸다 → 원인 = TT 학습 (조건 혼합 희석 + 학습량), 데이터 아님.
- horizon: 500 step 한 번에 = 실패 (1 step 오차가 s0 유지보다 큼). 원본처럼 짧은 window 가 필요 (h8 1 step 3.7°·1 cm). window 이어 붙이기 누적 원인 = 횟수 (stride 1/4/7 비교) → h32 가 관절 최선.
- 위치 오차의 주원인 = 몸통 yaw·속도 오차 (직진에서 없는 회전 생성). 위치 delta 표현 (C) 은 효과 없음. 조건을 dynamics 만으로 (G2) 하면 yaw 11.0°→5.9°, 위치 13.1→8.6 cm (MLP 와 격차 남음).
- 원본 TT 확인: 4 족 학습 task = `hardware_gen` 하나 (목표 궤적 → 형상·state·제어, ε 예측, 100 단계). 5 단계는 `ctrl` task 값 (이전 기록 정정). 목표 궤적 조건은 사용자 의도와 달라 쓰지 않는다.
- 코드 `code/tt_dynamics/` (train_tt.py, rollout_data.py, diag_mlp.py), 지표 `runs/pipeline_v1/tt/<run>/` (checkpoint·sample npz 는 로컬만).
- G1 (h32_try4, 3 조건, 100k step): 32 step yaw 5.6°·위치 8.4 cm·관절 6.8° = G2 수준 이상 → 희석은 학습량으로 메워진다. 남은 격차: 속도 0.153 vs MLP 0.055 m/s, 회전 모션 yaw 15.2°.

### 8-6. track_try1 진단 A (TT 예측 vs MJX, 32 step window 증분, 2026-10-18)
- 결론: dynamics 원인 확정. scale 0 회전 window yaw 오차 19.5° vs 직진 5.8° (3.4 배, 기준 2 배). TT 는 회전 Δyaw 13.5° 예측, MJX 실제 3.5°. yaw 방향 일치 55~65%.
- 직진 window 에서도 MJX 는 window 당 +3.7° 좌편향, TT 는 0.2° 예측. 코드 `code/tt_dynamics/diag_task.py`, 결과 `runs/pipeline_v1/task/track_try1/diag_task.json`.

### 8-7. zigzag_try1 + 형상 비교 (2026-10-18, 원형 대비 보상만 변경: 진폭 0.3 m·주기 2 m·반폭 0.15 m)
- 결론: 형상 E 가 task 에 반응하지 않는다. 원형 vs 지그재그 평균 E 거리 0.0013 (학습 형상 간 0.95~1.15), 전부 e000 근방. scale 10 예측 2.14 m vs MJX 0.41 m.
- 원인: s0 가 E 를 결정. s0 를 e001·e002 rollout 에서 가져오면 생성 E 가 각각 e001·e002 로 거리 0.007 이내. 학습 형상 3 개 → E = 3 점 식별.
- 보고서 `runs/pipeline_v1/report.html` (전 단계), 다음 후보 N1 s0 형상 중립화 / N2 형상 수 확대 / N3 회전 데이터 / N4 재계획.

### 8-8. h32_try5 (+free 조건, s0 도 생성) → track_try2·zigzag_try2 (2026-10-18, try4·try1 대비 mode 3 + `--free-s0` 만 변경)
- 결론: 기준 (두 task 평균 E 거리 > 0.1) 불통과, 0.037 (scale 10). E 는 e000·e001·e002 로 나뉘지만 후보별 형상은 노이즈가 정한다 (같은 seed → scale·task 무관 같은 형상). 예측-실제 격차는 감소 (지그재그 scale 10: 2.14 vs 0.41 m → 0.81 vs 0.76 m).
- TT 오차 try4 대비 약 10 % 증가 (32 step yaw 6.2°·위치 9.5 cm·관절 7.2°). scale 0 낙상 8/8 (3~186 step): 생성 s0 가 분포 밖 (높이 0.11~0.37 m vs 데이터 시작 0.28~0.32, 속도 최대 1.4 m/s) — free 조건이 episode 중간 window (낙상 중 포함) 의 시작 state 를 배웠기 때문.

### 8-9. 진단 D1 open-loop 민감도 (2026-10-18, 데이터의 실제 행동을 MJX 에 재생, TT 무관)
- 결론: open-loop 자체는 주원인 아님 (작은 교란 ±0.01 rad·±1 mm 에서 10 s 낙상 평균 23 %, 기준 50 %). 1 s 시점 위치 차이 0.4~0.7 cm 로 TT 32 step 오차 8.4 cm 보다 10 배 이상 작다 → 생성 실패의 주원인은 TT 예측 오차.
- 형상 차이: e000 낙상 0 %, e001 33 %, e002 36 % (교란 없는 대조군도 6·16 %). e001·e002 는 10 s open-loop 에서 불안정 → 긴 실행엔 재계획 필요. 코드 `code/tt_dynamics/diag_openloop.py`, 결과 `runs/pipeline_v1/diag/D1_openloop/`.

### 8-10. 진단 D2 데이터 분포 밖 여부 (2026-10-18, 생성 step 의 학습 데이터 최근접 거리)
- 결론: 새로운 물리 영역 아님. 생성 step 중 데이터 p99 거리 초과 0.07 % (데이터 자신 1 %, 기준 3 %), scale 10 도 0 %. 초과 비율과 예측-실제 차이 상관 −0.09 → TT 가 데이터 범위 안에서 틀린다 (TT 학습 문제, P4·P13).
- 단, 생성 step 의 최근접 거리 중앙값 0.21~0.32 = 데이터 (0.055) 의 4~6 배 → 데이터가 드문 사이 영역. step 단위 검사라 전이 (s, a → s') 의 오류는 못 본다 → D3. 코드 `code/tt_dynamics/diag_ood.py`, 결과 `runs/pipeline_v1/diag/D2_ood/`.

### 8-11. CGR 코드 대조: free 조건 불일치 수정
- `report.html`과 코드 대조에서 mode 3의 E 마스크 오류 확인: 학습에서는 정답 E가 주어지고 E loss가 0이지만, `--free-s0` 실행은 E도 숨겼다. try5/try2의 free 공동생성 해석은 재검증 필요.
- 학습·실행이 `tt_model.condition_masks`를 공유하도록 수정. mode 0/1/2 유지, mode 3의 E·s0 모두 미지. `conditioning_version=2` 기록, 구형 checkpoint의 free 실행은 차단.
- `h32_freefix_cgr`는 try5와 같은 100k·4조건 설정으로 오류만 바꾸는 비교 실험. 분석과 남은 병목은 `runs/pipeline_v1/ROOT_CAUSE_REVIEW.md`; 형상 3개, 첫 window에서만 E 결정, 초기 상태·dynamics 문제까지 해결됐다는 뜻은 아니다.
- 결과 (h32_freefix_cgr): 32 step yaw 6.17°·위치 10.4 cm (try5 와 동급). free-s0 원형 scale 0 유효 0.87 → 6.04 s 로 늘었지만 8/8 낙상 → 버그는 주원인 아님.

### 8-12. S 직접 guidance 끄기 → track_try3·zigzag_try3 (2026-10-19, try1 대비 `--no-guide-s` 만 변경, h32_try4·s0 고정·scale 1/3/10)
- 결론: 판정 B. 생성 S 와 (생성 E·A 로 TT 재예측한 S) 의 불일치 10~11.5 → 7.0~7.6 cm (= seed 흔들림 6.3~7.1 cm) 로 직접 경로 몫 제거 확인. 원형 경로 차 0.36~0.86 → 0.26~0.43 m.
- 그러나 MJX reward 6 중 5 하락, TT 예측 reward 도 하락: 직접 경로가 S→A 를 끌어주던 통로였고, 남은 ∂R/∂A 는 TT 의 A→S 가 느슨해 (같은 입력 seed 흔들림 6 cm) 약하다. on·off 모두 실제 진행 ≈ 0, E 거리 0.001.
- 코드 `generate_task.py --no-guide-s`, `diag_consistency.py`. 보고서 7-6. 병합 코드 재실행: TT 생성 비트 동일, MJX 는 GPU 실행 간 미세 차이 (평균 0.002, 낙상 ±1 step) → 후보 1 개 수준 reward 차이는 판정 근거 아님.

### 8-13. 진단 D3 생성 궤적 전이 오차 (2026-10-19, 생성 E·s0·A 를 window 단위로 MJX 재실행)
- 결론: 생성 궤적의 전이 오차 = 평소 TT 오차의 0.7~1.4 배 (위치), yaw 1~2.6 배. 생성이 특별히 나쁜 게 아니라 TT 의 A→S 정확도 자체가 주원인 (속도 오차 0.172 ≈ 가만히 있음 기준 0.180).
- 코드 `code/tt_dynamics/diag_transition.py`, 결과 `runs/pipeline_v1/diag/D3_transition/`.

### 8-14. s0 A안: mode 4 (rest-s0) → h32_try6 · track_try4 · zigzag_try4 (`--half-width 0.15`) · track_try4ctrl (2026-10-19)
- 결론: s0 해제가 E 고정 장벽이었다 (ctrl = try6 + s0 고정 → E std 0.002 로 다시 0.5). 단 생성 E 는 전부 학습 3 점 (e000·e001·e002) 중 하나와 거리 0.01 이내, 형상은 seed 가 정한다.
- 기준: s0 높이 범위 안 4~5/8 (미달), scale 0 낙상 4/8 (통과), E 거리 0.12 (후보 1 개 탓, 사실상 미달). h32_try6 diag_window 정지 yaw 6.14° (try4 4.51°). scale 10 track R: try1 −0.95, ctrl +0.34, try4 +0.19.

### 8-15. 진단 D4 형상 민감도 (2026-10-19, 같은 s0·A 에 E 만 교체, TT vs MJX Δ)
- 결론: 판정 B. 높이 Δ 만 맞고 (r 0.47~0.68, 기울기 ≈ 1), 전진·yaw·pitch Δ 는 TT 가 거의 0 (기울기 0.01~0.27). 다른 형상 E 를 넣으면 오차 2 배 (전진 6.2 → 12.4 cm). ckpt try4·try6 동일.
- 코드 `code/tt_dynamics/diag_shape.py`, 결과 `runs/pipeline_v1/diag/d4_try1{,_ckpt4}/`.

### 8-16. guidance gradient 진단 (2026-10-19, h32_try6, generate_task 첫 window 재현 PASS)
- 결론: diff_steps 5 에서 ᾱ = [0.899, 0.648, 0.341, 0.094, 0.0001], 첫 step 은 clamp 포화로 무의미. 형상은 k=3 에서 결정 (최종과 7/8 일치), 이후 network 가 학습점 쪽으로 0.1~0.3 당기고 guidance (0.09 → 0.007) 는 그보다 작다.
- token 당 |∇S| > |∇E| > |∇A| ("A 가 E 를 가림" 기각). guidance = 학습 분포 × exp(R) 이므로 step 을 늘려도 3 점 사이 형상은 안 나온다 → 형상 수 확대 필수. 코드 `diag_grad.py`, 결과 `tt/h32_try6/diag_grad.json`.

### 8-17. 진단 D5 행동 민감도 (2026-10-19, `diag_shape.py --vary A --offset 0.1`, 175 window)
- 결론: 가설 확인, TT 가 A 를 거의 무시. offset A 에 TT 반응 = MJX 실제 Δ 의 6~10 % (기울기 0.01~0.04), swap A 24~33 % (yaw 상관 0.02). 높이만 0.66.
- 원인 가설: 데이터 A = π(S) 라 A 가 중복 정보. E 도 같은 구조 (D4). 해결 후보 = 반사실 쌍 데이터 (같은 s0 에 A offset·swap·E swap 을 MJX open-loop 32 step, diag_shape mjx_run 재사용). 결과 `runs/pipeline_v1/diag/d5_try1/`.

### 8-18. 다음 계획: 재계획 실행 (2026-10-19, 사용자 우선순위 = 지금 나온 E 로 복합 작업부터 성공)
- 원인: `generate_task.py` 는 10 s 를 TT 상상으로 한 번에 생성 후 MJX open-loop 실행 → window 16 개에 TT 오차 누적 (실제 진행 ≈ 0).
- 계획 `--replan K` (K=16): E 는 첫 생성으로 고정, K step 실행 → MJX 실제 state 를 s0 로 mode 2 + guidance 재생성 반복. 단변수 = track_try4ctrl 대비 실행 방식만. 통과: scale 10 MJX 진행 ≥ TT 예측의 50 %, 낙상 ≤ 2/8. A 통과 → 형상 문제로 / B 미달 → 반사실 데이터 / C 무변화 → 실행기 (expert) 검토.
- GPU 0 은 HBM_paper2 A14 큐 (10-20 01:30 전후 종료 예상), 실험은 GPU 1 (P620).

### 8-19. action 반사실 A/B 다형상 평가 (2026-10-20)
- 결론: B는 학습 형상 e000/e001/e002의 held-out episode에서 관절 ΔMAE 6/6, yaw 5/6, 측면 위치 3/3 개선; 전진 위치는 3/3 소폭 악화. 결과 `runs/pipeline_v1/action_ablation/multishape_test_e{000,001,002}.json`.
- 단, B는 A보다 전체/학습 trajectory가 3배 (전체 1,728/576, train 1,200/400 episode)라 pairing 단독 효과는 분리 불가. 미학습 형상 일반화·복합 task 성공은 미검증. 상세표 `runs/pipeline_v1/action_ablation/README.md`.

### 8-20. D4 다중 형상 민감도 (frozen h32_multishape_B, 2026-10-20)
- 같은 held-out `qpos/qvel/A`를 e000/e001/e002에 그대로 적용, 추가 학습 없이 32 state·31 control MJX와 TT seed 8개를 평가. 유효 시작 199개 (source e000/e001/e002 = 6/192/1), 공통 초기 접촉까지 일치한 시작은 13개뿐.
- 판정: TT가 형상 변화에 일부 반응하지만 yaw·joint 변화 slope는 약 −0.024~0.065, 측면도 약함. e002−e000 전진 slope 0.70, 높이는 0.16~0.54로 일부 반응. 전체 요약은 source 불균형이 심해 대표성 제한.
- 원인/제약: 768개 held-out 후보/source를 audit했으나 초기 충돌·관절 한계·유한성·공통 action 조건을 모두 통과한 시작은 e000-source 6개, e001-source 718개, e002-source 1개뿐. 자세/높이를 보정하지 않는 조건에서 balanced physics-valid 공통 시작이 부족하다. `runs/pipeline_v1/diag/D4_morphology_v3/README.md` 및 JSON·NPZ·figure1~5 참조. 균형 있는 판정 전에는 임의 보정하지 말고 공통 유효 시작 확보 방안을 사용자와 결정한다.

### 8-21. D4 pairwise 재평가 (frozen h32_multishape_B, 2026-10-20)
- 세 형상 공통 교집합 대신 각 pair를 독립 audit했다. 같은 qpos/qvel/A와 두 형상 모두 유효·초기 4발 접촉 일치 조건으로 e000↔e001 15개(6/9), e000↔e002 21개(7/14), e001↔e002 18개(18/0; source e002는 0개)를 확보. 모든 후보를 검사했고 추가 학습 없음.
- 결과: 세 pair 모두 yaw TT RMS ≪ MJX (4.38/43.99°, 5.25/26.59°, 7.75/30.75°), yaw slope ≈0.006/−0.0003/−0.057. 관절 slope도 약 0.005~−0.010으로 거의 무반응. 높이는 부분 반응(기울기 0.31~0.47); 전진은 e000↔e002만 slope 0.61, 나머지는 약함/반대. 형상 민감도는 여전히 부족하다는 증거.
- 제약: 표본은 작고 source 편중, 특히 e001↔e002는 e001-source만 포함하므로 전체 source 분포 일반화는 주장하지 않는다. `runs/pipeline_v1/diag/D4_pairwise_v2/README.md`, pair별 `data.npz`·그림 참조.

### 8-22. 다음 S₀ 생성 진단 및 wide16 expert 재학습 (2026-10-20)
- 생성 E–S₀의 초기 물리 유효성·접촉·짧은 안정성을 확인하는 실험은 `runs/pipeline_v1/diag/S0_GENERATION_PLAN.md`에 별도로 기록하고 wide16 expert 재학습 이후 진행한다.

### 8-23. Foot-Anchored 연속 형상 파이프라인 구현 (2026-10-22)
- 실행 문서: `runs/pipeline_v1/foot_anchored_v1/README.md`, 단계 실행기: `code/tt_dynamics/run_foot_anchored_pipeline.sh`.
- `prepare`: 연속 E별 여러 S₀를 생성하고 4발 접촉·관절 한계·자체 충돌·지면 관통·지지 다각형·짧은 안정화를 검사한 뒤, 동일 E·S₀에서 Orig/Swap/Offset을 MJX로 수집한다. E 단위 train/val/test 분할을 사용한다.
- `dynamics`: Orig-only와 intervention Mode 1을 같은 정규화로 비교하고, 미학습 E의 D5 및 strict-S₀ 가능성 audit와 height-aligned D4를 기록한다.
- `joint`: mode 1+4로 E·S₀·S·A 공동 생성을 학습한다. 생성 S₀를 보정 없이 MJX에 넣어 4발 접촉·생존·경로 오차 gate를 통과해야 한다.
- `guidance`: gate 통과 뒤에만 미학습 zigzag reward guidance를 10초로 평가한다.
- 기존 wide16에서 관찰된 후반 정책 붕괴에 대응해 낮은 PPO 학습률과 학습 형상 기반 중간 체크포인트 선택을 적용한 재학습은 `runs/pipeline_v1/experts_wide16_v2/README.md` 참조. 기존 결과를 덮어쓰지 않는다.

### 8-24. e_rest_s0 smoke 렌더 + 근본 원인 진단 (2026-10-25, v4 ckpt, zigzag 2 s, 후보 4)
- 결론: gate 9/9 PASS 이나 scale 0 예측-MJX 경로 차 0.21~0.83 m, MJX R 전부 음수. 렌더 재실행(seed 재현 동일) `runs/pipeline_v1/fast_mode24_v4_e_rest_s0_try1/report.html`.
- 배제: 실행 parity 정상 (데이터 target_q 재생 오차 0, 0.03 rad 노이즈 2 s 오차 3~5 cm, 낙상 0). zigzag 는 데이터 분포 안 (19 모션). Mode 4 E loss 정상.
- 발견: home S0 가 데이터 분포 밖 (몸통 z p0~2 %, 관절 최근접 0.34~0.53 rad, 데이터에 관절 정지 0 %), guidance 가 생성 S 에 직접 gradient (S_l2 64 ≫ A_l2 13).

### 8-25. expert warm-up S0 (`--expert-warmup-s0`, 2026-10-25, 8-24 대비 S0 만 변경)
- 결론: S0 를 분포 안으로 넣어도 scale 0 경로 차 0.18~0.81 m 로 불변 → S0 는 주원인 아님. 8-24 의 "주원인" 판단 철회.
- 구현: multi-design expert 를 생성 E 위에서 closed-loop 로 걸려 0.5 s 이후 첫 균형 상태 (2발 이상 접지, 발 침투 ≤ 2 mm, 직립) 를 S0 로. reference = E 최근접 학습 형상. gate 기준 불변.
- try1 (1 s 끝 상태) 은 gate 2 mm 에 걸려 0/8 (데이터 걸음 상태도 30 % 가 2 mm 초과). try2 (균형 순간 선택) 유효 7/8. `runs/pipeline_v1/fast_mode24_v4_warmup_s0_try{1,2}/`.

### 8-26. 교차 재실행 데이터로 joint TT 재학습 (2026-10-25, 8-19 의 A안을 Mode 2/4 에 적용)
- 결론: 교차 재실행 효과 없음/악화. 후보 16 scale 0 경로 차 중앙값 v5_orig 0.20 m vs v5_paired 0.34 m, 완주 12/16 vs 9/16. 개선은 학습량 (v4 3000 → 20000 step, 평균 0.48 → 0.20 m) 효과.
- 조건: v5_orig = v4 데이터 (wide16 primitive 16 형상), v5_paired = + `wide16_action_shared` Orig/Swap/Offset. 둘 다 20000 step × batch 32, Mode 2/4. paired 는 유효 step 1.4 배라 단독 효과 분리 불가.
- v5_orig guidance 효과 있음 (MJX R 중앙값 −1.33 → −0.76, 완주 12 → 14). 예측-MJX R 차 중앙값 1.1~1.3 잔존. `runs/pipeline_v1/fast_mode24_v5_compare/report.html`.

### 8-27. 학습량 60000 step (2026-10-25, v5_orig 대비 step 만 변경) — 통과
- 결론: 후보 16 scale 0 경로 차 중앙값 0.20 → 0.05 m, 완주 12/16 → 16/16, 예측-MJX R 차 1.27 → 0.05. val loss 0.032 → 0.012. TT 부정확의 주원인 = 학습 부족.
- 주의: 이동 거리 중앙값 0.41 → 0.25 m (데이터 ~0.35 m) 로 덜 움직이나 상대 오차도 49 → 20 % 로 개선. zigzag progress 는 여전히 ~0 (예측도 ~0).
- 결과 `runs/pipeline_v1/fast_mode24_v6_steps60k/`, 비교 `runs/pipeline_v1/fast_mode24_v5_compare/report.html`.

### 8-28. 미해결 (2026-10-25)
- 병목이 TT 정확도 → guidance 로 이동: TT 가 정확해졌지만 guidance 가 전진 동작을 만들지 못함 (예측 progress ~0). action_rate penalty 가 progress 보다 큰지 등 reward 구성 점검 필요 (미검증).
- 10 s 단계에서는 window 단위 MJX 재계획 (8-18) 필요.

### 보고서 형식 선호 (2026-10-22)
- 이후 실험 보고서는 `runs/pipeline_v1/final/report.html`의 시각적 HTML 구성(요약, 탐색, 결과 표, 그래프, 해석과 한계)을 기본으로 사용한다. Markdown 보고서 대신 브라우저에서 볼 수 있는 `report.html`을 생성한다.
- Mode 2·4 FAST 학습 보고서: `runs/pipeline_v1/fast_mode24_v4/report.html`. 기존 `REPORT.md`는 사용자 요청으로 삭제했다.
