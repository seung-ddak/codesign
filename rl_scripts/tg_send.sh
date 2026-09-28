#!/bin/bash
# 최소 텔레그램 전송기. 한 줄 메시지만 보낸다 (로그 첨부 없음).
#
#   bash scripts/tg_send.sh "quad/foot_terrain 학습 완료"
#
# 자격증명: ~/.config/telegram_bot.env (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID)
set -u
[ $# -lt 1 ] && { echo "Usage: $0 \"<message>\"" >&2; exit 1; }

ENV_FILE="${TELEGRAM_ENV_FILE:-$HOME/.config/telegram_bot.env}"
[ -f "$ENV_FILE" ] || { echo "[tg] env 없음: $ENV_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"
[ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] || {
  echo "[tg] TOKEN/CHAT_ID 비어 있음" >&2; exit 1; }

curl -s -m 20 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  --data-urlencode "chat_id=${TELEGRAM_CHAT_ID}" \
  --data-urlencode "text=$1" >/dev/null \
  && echo "[tg] sent: $1" || { echo "[tg] 전송 실패" >&2; exit 1; }
