#!/usr/bin/env bash
# 7 단계 task 생성 → MJX 실행·녹화 → 결과 폴더를 같은 경로로 GitHub 에 commit·push.
#   bash code/tt_dynamics/run_task.sh <out 폴더> [generate_task.py 인자...]   (codesign/ 에서)
set -euo pipefail
cd "$(dirname "$0")/../.."
out=$1; shift
[[ ! -e "$out/summary.json" ]] || { echo "Existing result: $out" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1
../transformer-transformer/.venv/bin/python code/tt_dynamics/generate_task.py --out "$out" "$@"
git=(env -u LD_LIBRARY_PATH git)
"${git[@]}" diff --cached --quiet || { echo 'Unrelated staged changes; refusing commit' >&2; exit 1; }
"${git[@]}" add -- "$out" code/tt_dynamics
"${git[@]}" commit -q -m "Task result: $out

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
"${git[@]}" push origin HEAD:main
echo "Published $out $("${git[@]}" rev-parse --short HEAD)"
