# 생성 E·S₀ 지면 관통 원인 추적 — 단계별 중단 실험

목적: 기존 FAST Mode 2·4 체크포인트를 재학습하지 않고, **저장된 실제 MJX 상태와 생성된 E·S₀**를 동일한 MuJoCo 모델·변환 함수·물리 기준으로 비교한다. 기존 Smoke 결과를 덮어쓰지 않는다.

## 실행 명령 (학교 GPU 서버, CPU MuJoCo 검사)

    cd ~/quad_ws/codesign
    git fetch origin
    git switch experiment/wide16-experts
    git pull --ff-only

    ../transformer-transformer/.venv/bin/python -m pytest tests/test_s0_geometry.py tests/test_mode4_anchor.py

    ../transformer-transformer/.venv/bin/python code/tt_dynamics/diagnose_s0_penetration.py

입력: 원본 Wide16 MJX Rollout 중 e000/e001/e002 (기본 3개 형상), 실제 FAST Smoke의 scale_0/scale_1 NPZ, 동일한 walkForwardSlow expert config. 예시마다 Height를 자동 수정하거나 발 관통 허용치를 변경하지 않는다.

## 자동 실행 순서 (FAIL 즉시 종료)

- 0 **입력/설정 확인**: 필요한 NPZ와 Expert config가 있는지 확인한다. 실패하면 파일/환경 오류를 기록하고 종료한다.
- 1 **실제 데이터 직접 검사**: 각 원본 형상 E로 모델을 만든 후 원본 MJX qpos/qvel의 발/몸체 관통을 검사한다. 기본적으로 각 형상에서 정상 Frame 3개를 찾는다. 확보하지 못하면 모델 구성/원본 데이터/접촉 판정 문제부터 해결해야 하므로 여기서 종료한다.
- 2 **실제 데이터 → TT State → MuJoCo State 복원**: 1단계에서 통과한 실제 물리 상태의 S₀를 rollout_data.features로 만들고, 생성 코드가 사용하는 동일한 s0_geometry.s0_to_q로 복원한다. 발 높이 차이, S Token 차이, 물리 검사 실패가 있으면 여기서 종료한다.
- 3a **기존 Smoke 관통 결과 재현**: 저장된 scale 0과 scale 1의 E/S₀에서 발 높이를 다시 계산하고, 기존 summary.json에 기록된 관통 수치와 비교한다. 불일치하면 모델/실험 설정 차이일 수 있으므로 종료한다.
- 3b **생성 E·S₀ 물리 유효성 평가**: 위 단계들이 통과한 경우에 한해 생성 결과의 지면 관통을 판정한다. 실패하면 "원본 MJX 상태 및 변환은 통과했지만 생성 E/S₀는 물리적으로 불일치"라고 기록하고 종료한다. 이는 생성 모델의 어느 내부 부분이 문제인지를 확정한 뜻은 아니다.

## 결과 파일

- runs/pipeline_v1/s0_root_cause_audit/diagnosis.md : 현재 단계, 실패 이유, 후속 확인 대상
- runs/pipeline_v1/s0_root_cause_audit/diagnosis.json : 각 단계의 구체적 원자료, 유효 Frame 수, 발/비발 관통 깊이, 변환 오차, Guidance 0/1 후보별 결과

실패한 단계까지 보고서를 저장하고 **종료 코드 1**을 반환한다. 코드/환경 오류는 종료 코드 2다. 모든 물리 검사에 통과해야 종료 코드 0이 된다. 2초 MJX Open-loop stepping은 이 스크립트에서 수행하지 않으며, 별도 Smoke 단계에서 실행해야 한다.

## 선택 설정

    ../transformer-transformer/.venv/bin/python code/tt_dynamics/diagnose_s0_penetration.py \
      --sources e000 e001 e002 e004 \
      --probes-per-shape 3 \
      --max-probes-per-shape 90 \
      --out runs/pipeline_v1/s0_root_cause_audit_4shapes

기본 허용 기준은 지면/자체 관통 2 mm 이하, 발 접촉 최소 2개, 변환 전후 발 높이 차이 1 mm 이하다. 진단 자체가 실패했다는 이유로 수치를 완화하지 않는다.

### 해석 유의사항

- 원본 MJX Frame이 검사에서 실패하면 먼저 **모델 choice, 데이터, 접촉 기준**을 확인한다.
- 원본은 정상이나 복원 State가 실패하면 **인코딩·디코딩 경로**를 확인한다.
- 둘 다 정상이고 기존 Smoke까지 정확히 재현했으나 생성 E/S₀만 실패하면 **Joint Diffusion 결과의 물리 정합성**을 추가 조사한다.
- 원래 생성 결과는 Guide 0·1 한 후보씩이므로, 이번 실험만으로 생성 물리 유효율을 일반화할 수 없다.
