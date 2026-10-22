# Pipeline v1 FAST PATH — 목적 회복 및 실행 가이드 (2026-10-22)

## 첫 검증의 질문

기존 PPO expert가 MJX에서 만든 실제 (E, S, target_q, S_next)
연속 궤적만으로 학습한 TT가, 보지 않은 task의 **미분 가능한 reward**
로 **매 denoising step에 E·S·A의 노이즈를 동시에 guidance**하여 생성한
형상과 목표 관절각을 MJX에서 실행할 수 있는가?

검증 전에 형상 변화 D4의 모든 slope가 1일 필요는 없다.
그렇다고 model error가 무관한 것은 아니다. 생성 E·A의 **실제** MJX
reward, 유효한 초기 상태, 완주 여부가 첫 성공 판정 기준이다.

## 어디서 병목이 시작됐나? (repo 자료에 근거)

- 10/17 `runs/pipeline_v1/ROLLOUT_PLAN.md`: 19 expert, **e000 하나**
  10초 전환 데이터를 먼저 수집하여 *파이프라인 작동을 확인*.
  256-episode 수집 중 41.4% 낙상 → 실제 데이터 품질 이슈.
- 10/17–18 `remind.md` 8-5~8-8: TT dynamics 학습/회전 오차/생성 E의
  s0 종속성과 free-s0 낙상 진단으로 복합 task 실행이 어려워짐.
- 10/19 `remind.md` 8-11~8-18: 학습·추론 conditioning mask 불일치 수정
  (필수 정확성 문제). D3/D4/D5/gradient를 추가했으나,
  8-18에는 이미 사용자 우선순위가 **현재 나온 E로 복합 작업을 먼저 성공**
  이라고 명시됨.
- 10/20~22 D4 → 16 shapes → counterfactual action/morphology intervention
  → Foot-Anchored v2 → 엄격한 동적 gate의 직렬 선행 조건으로 확대.
  이 지점에서 개별 E sensitivity 진단이 사실상 프로젝트 전체의
  통과 조건이 되었고 최초 빠른 검증 목적과 어긋났다.

특히 D4의 **고정 qpos/qvel/A**는 형상별로 최적화된 A/S0가 다를 수 있는
Joint Co-design을 그대로 재현하는 실험이 아니다. 진단으로 유지한다.

## FAST 필수 경로 (4단계)

1. **기존 수집 데이터 이용:** 실제 MJX에서 기록한 primitive/transition
   rollouts를 그대로 사용. PPO 재학습, 새 Foot-Anchored, D4 pair 생성,
   16개 전체 재수집을 우선 진행하지 않는다.
2. **작은 TT joint training:** 기존 `train_tt.py`의 H=32,
   `--modes 2 4`, DDIM 5로 작은 smoke checkpoint 작성.\n   첫 Window는 Mode 4에서 E·S₀·S·A 생성, 다음 Window부터\n   Mode 2에서 E·S₀ 조건으로 S·A 생성. Mode 1은 초기 FAST에 불필요.
   **Reward는 TT pretraining에 넣지 않는다.**
3. **Reward-guided joint diffusion:** `--joint-full-guidance`가
   `joint_diffusion_guidance.py`를 호출. 각 reverse diffusion step에서
   모든 window의 예측 10초 E·S·A를 결합해 task reward를 계산한다.
   `torch.autograd.grad`로 **E 노이즈에도 gradient를 전달**하고
   S/A 노이즈와 함께 denoise한다. E는 모든 window에서 하나로 공유된다.
   이전 SPSA 외부 E 후처리는 사용하지 않는다.
4. **MJX 검증:** 동일한 Diffusion 시드의 scale 0 대 guided scale 1을
   실제 실행해서 reward, 낙상, 초기 S0, E-gradient nonzero를 비교한다.
   0.02초 간격의 Open-loop 목표 관절각 A만 실행한다.

### 왜 smoke와 full을 나누는가?

- `smoke` 2초는 배선/미분/물리 초기화의 소규모 기능 검증.
- `full` 10초는 **실제 복합 task 검증**. 2초는 최종 연구 성공이 아님.
- TT 모델이 5-step로 충분히 수렴했는지, 전환 동역학을 학습했는지는
  smoke에서 보장하지 못한다. 필요할 때만 학습량/데이터를 늘린다.

## 실행 명령 (학교 서버에서 사용자가 직접 실행)

```bash
cd ~/quad_ws/codesign
git fetch origin
git switch experiment/cp-design-generalization
../transformer-transformer/.venv/bin/python -m pytest \
  tests/test_cp_task_reward.py tests/test_joint_diffusion_guidance.py

# 3개 형상의 기존 데이터로 작은 smoke 모델 (기존 ckpt 있으면 학습 생략)
QUICK_STEPS=3000 bash code/tt_dynamics/run_quick_validation.sh train

# 2초: 형태 E 노이즈에 실제 gradient가 흐르는지 + MJX 실행 확인
bash code/tt_dynamics/run_quick_validation.sh smoke

# 10초 전체 reward 기반 공동 생성 + MJX 비교
bash code/tt_dynamics/run_quick_validation.sh full
```

16개 형상에서 이미 MJX로 수집한 `wide16_action`이 있다면
**재수집하지 않고** 다음과 같이 기존 데이터를 입력으로 선택할 수 있다.

```bash
QUICK_DATA_DIR=runs/pipeline_v1/wide16_action/rollouts \
QUICK_MAX_FILES=16 QUICK_STEPS=3000 \
QUICK_ROOT=runs/pipeline_v1/quick_joint_wide16 \
bash code/tt_dynamics/run_quick_validation.sh train

QUICK_ROOT=runs/pipeline_v1/quick_joint_wide16 \
bash code/tt_dynamics/run_quick_validation.sh smoke
```

재학습 없이 이전 Mode 2 + Mode 4 checkpoint를 시험한다면
`QUICK_CKPT=<existing compatible checkpoint dir>`를 지정한다.
실행용 Expert config 경로는 `QUICK_EXPERT_CONFIG`로 지정할 수 있다.
예전 100k checkpoint도 모델 조건이 맞으면 재사용할 수 있지만,
실제 파일이 이 머신에 있는지 확인해야 한다.

`QUICK_CANDIDATES=1` 기본값으로 GPU 비용을 최소화했고,
예상 score가 불안정하면 scale을 낮춘다.
출력은 별도 `runs/pipeline_v1/quick_joint/`에만 기록하며
기존 run/checkpoint를 덮어쓰지 않는다.

학습 로그의 `val_loss`는 Mode 2·4 혼합 검증 손실이다.
과거 `train_tt.py`의 학습 후 `eval_dynamics`는 Mode 1을 사용하므로,
Mode 1을 학습하지 않는 FAST에서는 자동 평가를 건너뛴다.
Joint 생성과 실제 MJX 재생으로 검증한다. `--modes 2 4`는
필요한 조건의 수를 줄일 뿐, E에 대한 물리적인 인과 모델이
완성되었음을 보장하지 않는다.

## 실패 시 한 가지만 수정한다

- 데이터/Checkpoint 누락 → 이전 서버의 기존 Rollout/Checkpoint 복사.
- Gradient가 E로 0 → 조건 마스크, e-token 경로, clamp, 생성 코드 점검.
- 생성 초기 S0 invalid → 생성 시작 상태 학습 분포 수정. Mode 4는
  정지·4발접지 S0를 조건으로 주지만, 기존 데이터에는 움직이는 RSI
  시작 상태도 있으므로 **Mode 4 학습/생성 조건 불일치 가능성**이 있다.
- S0 정상, MJX 움직임 불일치 → 1~32 state A→S Dynamics 확인(D5).
- 실제 Reward가 예측보다 매우 낮음 → TT 예측 오차/Reward hacking 확인.
- Task별 E 차이 없음 → 우선 서로 분리된 Reward와 여러 seed에서
  *실제 reward 개선*을 확인; 이후 형상 지원 범위·D4를 필요시에만 점검.
- 기본 동작이나 전환이 충분하지 않음 → 새 expert부터 학습하지 말고
  기존 16개 형상·3모션의 서로 다른 시작/속도/전환 rollout을 확대.
  여전히 부족한 모션만 Expert 학습.

## 성공 기록

1. Full-horizon denoising 단계에서 `e_gradient_nonzero=True`.
2. 생성 E·S0·A가 finite이며 S0 물리 감사 통과.
3. MJX open-loop 생존율, 실제 Reward, TT 예측-vs-실행 차이.
4. scale 0 vs guided의 공정한 같은-noise 비교.
5. 추가 단계에서 seed/Task/형상별 반복 및 **각 E에 적합한 A**
   재생성 후 Task별 교차 평가.

`ZigzagReward`는 현재 **중심선 경로 추종**으로, 원래 계획의
**기둥을 순서대로 통과하는 slalom**을 정확히 구현한 것은 아니다.
지금은 reward/gradient/실행 배선을 빠르게 검증하는 proxy이며,
이를 실제 장애물 회피 성공이라고 부르면 안 된다.

첫 성공 뒤 확장할 항목: 실제 slalom 기둥/충돌 reward, 10초 성능
재현성, 16개 형상 범위, Expert 전환 다양성, E 별 목적함수 rank.
이는 **별도 후속**이며 FAST 경로를 차단하지 않는다.

## 기존 코드 보존

`experiment/wide16-experts`와 Foot-Anchored v2/D4/D5/extra SPSA
진단 코드는 변경/삭제하지 않는다.
FAST 경로는 기존 복잡한 stage gate를 우회해 **작동 가능성만 확인**
하는 독립 실행기다. 예측 Reward의 상승만으로 성공을 선언하지 않는다.
