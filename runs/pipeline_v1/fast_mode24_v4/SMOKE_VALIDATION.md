# FAST Mode 2·4: 2초 E·S·A 공동 생성 검증

대상: `experiment/wide16-experts` 브랜치의 기존 체크포인트
`runs/pipeline_v1/fast_mode24_v4/tt_joint/checkpoint.pt`.
TT 재학습이나 PPO 재학습은 필요하지 않다.

## 무엇을 검사하는가?

1. Reward Gradient가 활성 형상 E, 상태 S, 목표 관절각 A로 각각 전달되는가?
   Denoising 각 단계에서 **분리된 Gradient norm**을 기록한다. 비활성 E 좌표는 의도적으로 제외한다.
2. TT가 같은 Random Seed에서 Guidance 0과 1로 2초 길이 E·S·A를 실제 생성하는가?
   출력 NPZ의 유한성, shape 및 두 생성 결과의 E/S/A 변화량을 확인한다.
3. 생성한 E와 S₀가 CPU MuJoCo/MJX 초기 물리 검사에 통과하는가?
   검사를 통과하지 못한 후보는 Open-loop stepping에서 제외하고 `mjx_total=null`로 기록한다.
4. 초기 상태가 유효한 후보가 2초 동안 MJX Open-loop 실행을 완료하는가?
   낙상, 실제 유효 시간, 실제 Reward를 확인한다.

단위 테스트 통과는 2초 실제 MJX 검증 통과가 아니다.

## 실행 (학교 GPU 서버)

```bash
cd ~/quad_ws/codesign
git fetch origin
git switch experiment/wide16-experts
git pull --ff-only

../transformer-transformer/.venv/bin/python -m pytest \
  tests/test_joint_diffusion_guidance.py \
  tests/test_mode4_anchor.py \
  tests/test_fast_smoke_audit.py \
  tests/test_cp_task_reward.py

QUICK_CKPT=runs/pipeline_v1/fast_mode24_v4/tt_joint \
QUICK_ROOT=runs/pipeline_v1/fast_mode24_v4_smoke_audit \
bash code/tt_dynamics/run_quick_validation.sh smoke
```

기본값은 Zigzag, 2초, 후보 1개, Guidance Scale 0과 1, 영상 생성 끔이다.
다른 실험을 할 때는 `QUICK_TASK=straight`, `QUICK_CANDIDATES=4`,
`QUICK_GUIDANCE_SCALE=0.5`와 같이 환경변수를 사용할 수 있다.

## 결과

- `smoke_zigzag_joint/summary.json`: 후보별 E, 실제 MJX Reward, 낙상,
  초기 물리 검사, 실행 제외 사유.
- `smoke_zigzag_joint/scale_0.npz`, `scale_1.npz`: 같은 Seed의
  Guidance 전/후 E·S·A와 실제 MJX 궤적.
- `smoke_zigzag_joint/smoke_check.json`: 각 검증 조건의 PASS/FAIL,
  E/S/A 변화량, 시작 상태 유효 후보 수, 완주한 MJX 후보 수.
- `summary.json["joint_full_guidance_audit"]["scale_1"]`:
  E/S/A Gradient norm 목록과 nonzero flag.

`run_quick_validation.sh smoke`는 요건 중 하나라도 실패하면
`smoke_check.json`을 기록한 뒤 **종료 코드 1**을 반환한다.
이것은 실험 실패 판정이지, TT 학습 체크포인트 손상을 의미하지 않는다.

### 유의할 점

- "Gradient nonzero"는 TT 그래프에서 Guidance 신호가 전달된다는 뜻이며
  E가 MJX 실제 움직임에 인과적으로 영향을 준다는 증명은 아니다.
- 초기 물리 검사는 최소 2발 접촉·관절 한계·관통·몸체 자세 등
  제한된 정적 검사다. 통과해도 동역학적으로 안정할 보장은 없다.
- 본 단계는 학습한 16개 형상/19종 Primitive의 2초 기능 검증이다.
  10초 Reward 향상이나 미학습 형상 일반화 판정이 아니다.
- 기존 출력 폴더가 존재하면 결과 덮어쓰기를 막는다.
  재실행할 때는 다른 `QUICK_ROOT`로 지정한다.
