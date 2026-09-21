#!/bin/bash
# remind.md §4.3 : pi_theta 학습이 끝나면 L0~L3 을 순차 실행한다.
#
#   bash scripts/run_L_stages.sh <pi_theta tmux 세션> <run_dir> [budget]
#
# L0 은 게이트다. prior 성공률이 전부 0 이거나 이미 포화(>0.5)면 지형 규격을
# 먼저 고쳐야 하므로 L1 이후를 돌리지 않고 멈춘다 (§4.3 L0 목적).
set -u

TT_DIR="/mnt/gstore/home/xiangyue/RL/transformer-transformer"
RL_DIR="/mnt/gstore/home/xiangyue/RL"
SESSION="${1:-stair_pi_theta}"
RUN_DIR="${2:-}"
BUDGET="${3:-4096}"
SEEDS="${SEEDS:-3}"
OUT="${TT_DIR}/design_search_results.jsonl"
PY="${TT_DIR}/.venv/bin/python"

notify() { bash "${RL_DIR}/scripts/tg_send.sh" "$1" >/dev/null 2>&1 || true; }

cd "$TT_DIR"

echo "[chain] waiting for tmux session '${SESSION}' to finish"
while tmux has-session -t "$SESSION" 2>/dev/null; do command sleep 60; done
echo "[chain] training session ended"

if [ -z "$RUN_DIR" ]; then
  RUN_DIR="$(ls -td ${TT_DIR}/wandb/offline-run-*/files | head -1)"
fi
echo "[chain] run_dir=${RUN_DIR} budget=${BUDGET} seeds=${SEEDS}"

run_stage() {
  local stage="$1"
  echo "[chain] === ${stage} ==="
  notify "${stage} 시작"
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
    "$PY" scripts/design_search_stairs.py \
      --stage "$stage" --run-dir "$RUN_DIR" --budget "$BUDGET" \
      --num-seeds "$SEEDS" --out "$OUT"
  local rc=$?
  if [ $rc -ne 0 ]; then
    notify "${stage} 중단 (exit ${rc})"
    exit $rc
  fi
  notify "${stage} 완료"
}

run_stage l0

# L0 게이트
read -r GT0 MEAN <<<"$("$PY" - "$OUT" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
row = [r for r in rows if r["stage"] == "L0"][-1]
print(row["success_gt0_frac"], row["success_mean"])
EOF
)"
echo "[chain] L0 gate: success_gt0_frac=${GT0} success_mean=${MEAN}"
STOP=$("$PY" -c "g=float('${GT0}'); m=float('${MEAN}'); print(1 if (g==0.0 or m>0.5) else 0)")
if [ "$STOP" = "1" ]; then
  notify "L0 게이트 정지 — 성공률 분포가 포화/전무 (gt0=${GT0}, mean=${MEAN}). 지형 규격 조정 필요"
  echo "[chain] stopping before L1"
  exit 0
fi

for stage in l1 l2 l3; do
  run_stage "$stage"
done

notify "L0~L3 전부 완료"
echo "[chain] done"
