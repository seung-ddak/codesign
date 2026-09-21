#!/bin/bash
# remind.md §4.3 : L0 -> L1 -> L2 -> L3 순차 실행 (T1 보행, TT ctrl 평가기).
# 네 단 모두 같은 예산·같은 성공 판정·같은 평가기를 쓴다.
set -u
TT="${CODESIGN_TT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/transformer-transformer}"
RL="${CODESIGN_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$TT"

BUDGET="${1:-256}"
OUT="${TT}/design_search_walk.jsonl"
notify() { bash "${RL}/scripts/tg_send.sh" "$1" >/dev/null 2>&1 || true; }

COMMON="--budget ${BUDGET} --n-envs 32 --episode-steps 600 --goal 0.8 \
        --traj 1 --inference-steps 2 --out ${OUT}"

for stage in l0 l1 l2 l3; do
  echo "=== ${stage} 시작 $(date '+%H:%M:%S') ==="
  notify "설계탐색 ${stage} 시작"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/design_search_walk.py \
    --stage "$stage" $COMMON 2>&1 | grep -aviE \
    "glfw|warn|Gym has|gymnasium|migration|Users of|See the|INFO"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    notify "설계탐색 ${stage} 중단 (exit ${rc})"
    exit $rc
  fi

  # L0 게이트: 성공률이 포화(>0.9)거나 전무(0)면 난이도를 먼저 고쳐야 한다
  if [ "$stage" = "l0" ]; then
    STOP=$(.venv/bin/python - "$OUT" <<'EOF'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1]) if '"L0"' in l][-1]
m = r["success_mean"]
print(1 if (m == 0.0 or m > 0.9) else 0)
EOF
)
    if [ "$STOP" = "1" ]; then
      notify "L0 게이트 정지 — 성공률 포화/전무. 목표 거리 조정 필요"
      echo "[gate] L0 성공률이 변별 구간 밖이다. 중단."
      exit 0
    fi
  fi
done

notify "설계탐색 L0~L3 전부 완료"
echo "=== 전부 완료 $(date '+%H:%M:%S') ==="
