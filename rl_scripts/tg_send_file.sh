#!/bin/bash
# 텔레그램 파일 전송 (RL 봇, 최승훈). 캡션은 한 줄.
#
#   bash scripts/tg_send_file.sh <file> "<한 줄 캡션>"
#
# .mp4 는 sendVideo, 나머지는 sendDocument. 자격증명: ~/.config/telegram_bot.env
set -u
[ $# -lt 2 ] && { echo "Usage: $0 <file> \"<caption>\"" >&2; exit 1; }
FILE=$1 CAPTION=$2
[ -f "$FILE" ] || { echo "[tg] 파일 없음: $FILE" >&2; exit 1; }
ENV_FILE="${TELEGRAM_ENV_FILE:-$HOME/.config/telegram_bot.env}"
[ -f "$ENV_FILE" ] || { echo "[tg] env 없음: $ENV_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$ENV_FILE"
[ -n "${TELEGRAM_BOT_TOKEN:-}" ] && [ -n "${TELEGRAM_CHAT_ID:-}" ] || {
  echo "[tg] TOKEN/CHAT_ID 비어 있음" >&2; exit 1; }

case "$FILE" in *.mp4) METHOD=sendVideo FIELD=video ;; *) METHOD=sendDocument FIELD=document ;; esac
curl -s -m 120 -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/${METHOD}" \
  -F "chat_id=${TELEGRAM_CHAT_ID}" -F "caption=${CAPTION}" -F "${FIELD}=@${FILE}" | grep -q '"ok":true' \
  && echo "[tg] sent file: $FILE" || { echo "[tg] 파일 전송 실패: $FILE" >&2; exit 1; }
