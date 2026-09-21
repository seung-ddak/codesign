# codesign 경로 설정 — 새 머신에서 한 번만 맞춘다.
#   source env.sh
# 기본값: 이 저장소가 <어디든>/codesign 이고, TT 레포가 그 옆 <어디든>/transformer-transformer 라고 본다.
export CODESIGN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CODESIGN_TT="${CODESIGN_TT:-$(dirname "$CODESIGN_ROOT")/transformer-transformer}"
echo "CODESIGN_ROOT=$CODESIGN_ROOT"
echo "CODESIGN_TT=$CODESIGN_TT"
[ -d "$CODESIGN_TT" ] || echo "  (경고) TT 레포가 없다 — SETUP.md 1 단계부터"
