#!/bin/bash
# remind.md §4.3 : L0 은 "잘 하는 정책"이 아니라 "형상을 변별하는 정책"을 찾는 단계다.
# 학습이 끝나기를 기다리지 않고, 체크포인트가 나오는 대로 작은 budget 으로 L0 을 돌려
# 성공률 분포가 가장 변별적인 지점을 고른다.
#
#   bash scripts/l0_checkpoint_sweep.sh <run_dir> "<목표 step 목록>" [budget] [seeds]
set -u

TT_DIR="${CODESIGN_TT}"
RL_DIR="${CODESIGN_ROOT}"
RUN_DIR="${1:-}"
TARGETS="${2:-40000000 100000000 200000000}"
BUDGET="${3:-512}"
SEEDS="${4:-2}"
OUT="${TT_DIR}/l0_sweep.jsonl"
PY="${TT_DIR}/.venv/bin/python"

notify() { bash "${RL_DIR}/scripts/tg_send.sh" "$1" >/dev/null 2>&1 || true; }
cd "$TT_DIR"

if [ -z "$RUN_DIR" ]; then
  # 학습 시작 직후에는 새 wandb 디렉터리가 아직 없어 이전 run 을 잡을 수 있다.
  # 기다렸다가 학습 세션보다 나중에 만들어진 디렉터리만 고른다.
  for _ in $(seq 1 60); do
    RUN_DIR="$(ls -td ${TT_DIR}/wandb/offline-run-*/files 2>/dev/null | head -1)"
    [ -n "$RUN_DIR" ] && [ -f "${RUN_DIR}/config.pkl" ] && break
    command sleep 10
  done
  echo "[sweep] auto-detected run_dir (확인 필요): ${RUN_DIR}"
fi
echo "[sweep] run_dir=${RUN_DIR} targets=${TARGETS} budget=${BUDGET} seeds=${SEEDS}"

for target in $TARGETS; do
  echo "[sweep] waiting for a checkpoint >= ${target}"
  ckpt=""
  while [ -z "$ckpt" ]; do
    ckpt=$(ls "${RUN_DIR}/checkpoints" 2>/dev/null \
      | grep -E '^[0-9]+$' \
      | awk -v t="$target" '$1>=t' | sort -n | head -1)
    if [ -z "$ckpt" ]; then
      # 학습이 죽었으면 가진 것 중 가장 큰 체크포인트로 끝낸다
      if ! tmux has-session -t stair_pi_theta 2>/dev/null; then
        echo "[sweep] training session gone; using the largest checkpoint"
        ckpt=$(ls "${RUN_DIR}/checkpoints" | grep -E '^[0-9]+$' | sort -n | tail -1)
        [ -z "$ckpt" ] && exit 1
      else
        command sleep 60
      fi
    fi
  done

  echo "[sweep] === L0 @ ${ckpt} ==="
  XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
    "$PY" scripts/design_search_stairs.py \
      --stage l0 --run-dir "$RUN_DIR" --ckpt "${RUN_DIR}/checkpoints/${ckpt}" \
      --budget "$BUDGET" --num-seeds "$SEEDS" --max-batch 2048 --out "$OUT" \
    || { notify "L0 sweep 중단 (@${ckpt})"; exit 1; }
done

echo "[sweep] summary"
"$PY" - "$OUT" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'checkpoint':>12} {'mean':>7} {'max':>7} {'>0 frac':>8} {'fell':>7}")
best, best_score = None, -1
for r in rows:
    ck = r["checkpoint"].rstrip("/").split("/")[-1]
    # 변별력 = 성공한 설계가 있으면서 포화되지 않은 정도
    score = r["success_gt0_frac"] * (1 - r["success_gt0_frac"]) * 4
    print(f"{ck:>12} {r['success_mean']:7.3f} {r['success_max']:7.3f} "
          f"{r['success_gt0_frac']:8.3f} {r['fell_mean']:7.3f}")
    if score > best_score:
        best, best_score = ck, score
print(f"\n가장 변별적인 체크포인트: {best} (score {best_score:.3f})")
EOF
notify "L0 sweep 완료"
