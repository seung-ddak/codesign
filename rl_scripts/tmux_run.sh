#!/bin/bash
# RL 폴더 공통 학습 런처. tmux detached 세션에서 명령을 돌리고, 종료되면
# 무조건 텔레그램 알림을 보낸다 (성공/실패 모두). walk_N 계열 전용
# start_walk_train.sh 과 달리 어떤 학습 명령에도 쓸 수 있다.
#
#   bash scripts/tmux_run.sh <세션이름> "<명령>"
#   bash scripts/tmux_run.sh codesign "source quad/activate.sh && python train.py --headless"
#
# 옵션 (환경변수)
#   TMUX_RUN_LOG=<경로>    콘솔 로그 tee 위치 (기본: RL/logs/tmux_run/<세션>_<시각>.log)
#   TMUX_RUN_CWD=<경로>    작업 디렉터리 (기본: RL)
#   TMUX_RUN_LABEL=<문구>  알림에 쓰는 학습 이름 (기본: 세션이름)
#   TMUX_RUN_NOTIFY=0      알림 끄기
set -u
[ $# -lt 2 ] && { echo "Usage: $0 <session> \"<command>\"" >&2; exit 1; }

RL_DIR="${RL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"  # scripts/ 의 부모 (codesign 옆)
SESSION="$1"; shift
CMD="$*"
CWD="${TMUX_RUN_CWD:-$RL_DIR}"
LABEL="${TMUX_RUN_LABEL:-$SESSION}"
NOTIFY="${TMUX_RUN_NOTIFY:-1}"
STAMP="$(date '+%Y%m%d_%H%M%S')"
LOG="${TMUX_RUN_LOG:-${RL_DIR}/logs/tmux_run/${SESSION}_${STAMP}.log}"

tmux has-session -t "=$SESSION" 2>/dev/null && {
  echo "ERROR: tmux 세션 '${SESSION}' 이 이미 있다. 먼저 종료할 것." >&2; exit 1; }

mkdir -p "$(dirname "$LOG")"

# 시작 알림 (사용자 지시: 시작/완료 둘 다 필수)
[ "$NOTIFY" = "1" ] && bash "${RL_DIR}/scripts/tg_send.sh" "${LABEL} 학습 시작" || true

# 종료 알림을 같은 셸 안에서 chain 한다. 별도 세션 감시(polling)는 쓰지 않는다.
# rc 는 tee 가 아니라 학습 프로세스의 코드여야 하므로 PIPESTATUS[0] 을 쓴다.
NOTIFY_TAIL=":"
if [ "$NOTIFY" = "1" ]; then
  NOTIFY_TAIL="if [ \$rc -eq 0 ]; then bash '${RL_DIR}/scripts/tg_send.sh' '${LABEL} 학습 완료'; else bash '${RL_DIR}/scripts/tg_send.sh' \"${LABEL} 학습 중단 (exit \$rc)\"; fi"
fi

tmux new-session -d -s "$SESSION" bash -c \
  "cd '${CWD}' && PYTHONUNBUFFERED=1 ${CMD} 2>&1 | tee '${LOG}'; rc=\${PIPESTATUS[0]}; ${NOTIFY_TAIL}"

echo "session : ${SESSION}"
echo "log     : ${LOG}"
echo "확인    : tmux capture-pane -pt ${SESSION} -S -40   |   tail -f ${LOG}"
