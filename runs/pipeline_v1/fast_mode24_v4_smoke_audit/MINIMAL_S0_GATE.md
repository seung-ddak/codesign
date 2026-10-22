# FAST 2초 S₀ 문제 — 최소 파이프라인 검증

목표는 TT 구조·Reward 전체를 바꾸지 않고 **클램프 여부를 단 한 번 확인한 뒤, 정상적인 생성 후보로 2초 MJX를 실행할 수 있는지** 보는 것이다. 이 검증은 물리 최적화나 10초 Task 검증이 아니다.

## 실행

```bash
cd ~/quad_ws/codesign
git fetch origin
git switch experiment/wide16-experts
git pull --ff-only

../transformer-transformer/.venv/bin/python -m pytest tests/test_s0_height_clamp.py tests/test_fast_smoke_audit.py

# 기존 TT 체크포인트 사용, 재학습 없음
QUICK_ROOT=runs/pipeline_v1/fast_mode24_v4_s0_gate_try1 \
QUICK_CANDIDATES=4 \
bash code/tt_dynamics/run_minimal_s0_gate.sh
```

## 순차 검증

1. **기존 실패의 S₀ 높이 Clamp 확인**: 체크포인트의 정규화 중심/범위를 읽어서 저장된 두 생성 결과의 초기 높이가 `-clip` 경계에 걸렸는지 기록. 같은 높이라고 무조건 Clamp로 판정하지 않음. 오류가 있으면 중단.
2. **일회성 제한 후보 검사**: 고정된 TT 체크포인트로 각 Guidance Scale(0,1)에 최대 4개 생성 후보를 같은 Seed에서 생성. 원래 생성한 E·S₀를 물리적으로 확인하고, 통과한 후보에서만 MJX 2초 Open-loop 실행. 기존 `run_quick_validation.sh smoke`의 PASS/FAIL 판정 사용. 미통과면 추가 자동 반복·재학습 없이 중단.
3. **새 결과 Clamp 재검사**: 성공/실패와 무관하게 새로 생성한 S₀가 또 경계에 걸렸는지 기록. 물리 Gate를 통과한 후보가 없으면 중단.

## 결과

`QUICK_ROOT` 아래에:
- `previous_s0_clamp.json`: 기존 Guidance 0,1의 초기 높이와 정규화 하한 비교
- `smoke_zigzag_joint/summary.json`: 생성 후보, 물리 유효성, MJX 실행/제외 이유
- `smoke_zigzag_joint/smoke_check.json`: Gradient 전달, E·S·A 공동 생성, 유효 S₀, 2초 실행 PASS/FAIL
- `new_s0_clamp.json`: 새 후보들이 정규화 하한에 걸렸는지 확인

**고정 조건:** TT 재학습 없음, S₀ 높이 인위적 이동 없음, 관통 허용치 완화 없음, 2초 Open-loop만, 실패 시 10초 단계로 넘어가지 않음.

주의: 기존과 새 결과가 계속 하한에 고정돼 있다면, 이 Gate 자체는 **Clamp 원인을 고치는 알고리즘이 아니다.** 한정된 Sampling으로도 유효 상태가 안 나오면 정확한 원인과 2초 실패를 기록하고 멈춘다. 그때부터 Clamp를 무작정 넓히거나 형상 설계 공간을 세밀하게 조정하지 말고, 생성 방식의 최소 수정이 정말 필요한지 판단한다. 이 방식은 검증 범위를 넓히지 않으면서 우선 실제 MJX 실행 가능성을 확인하는 단계다.
