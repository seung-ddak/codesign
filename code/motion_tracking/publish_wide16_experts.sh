#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
root=runs/pipeline_v1/experts_wide16
published="$root/github_published.txt"
branch=experiment/wide16-experts
[[ $(git branch --show-current) == "$branch" ]] || { echo "Expected branch $branch" >&2; exit 1; }

while true; do
  [[ $(git branch --show-current) == "$branch" ]] || { echo "Branch changed; stopping publisher" >&2; exit 1; }
  while IFS= read -r line; do
    [[ $line =~ ^Completed[[:space:]]([[:alnum:]_]+):[[:space:]](.*/motion_eval/summary\.csv)$ ]] || continue
    motion=${BASH_REMATCH[1]}
    summary=${BASH_REMATCH[2]}
    grep -qx "$motion" "$published" 2>/dev/null && continue
    [[ $summary == "$root/$motion/"* ]] || { echo "Unexpected summary: $summary" >&2; exit 1; }
    latest=${summary%/motion_eval/summary.csv}
    step=$(find "$latest/files/checkpoints" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tail -1)
    [[ -n $step && -f $latest/files/checkpoints/$step/_CHECKPOINT_METADATA && -f $summary ]] || exit 1
    git diff --cached --quiet || { echo "Git index contains unrelated changes" >&2; exit 1; }
    git add -- "$root/$motion/train.log" "$root/$motion/eval.log" "$root/queue.log"
    [[ ! -f $root/$motion/resume.log ]] || git add -- "$root/$motion/resume.log"
    git add -f -- "$latest/files/config.pkl" "$latest/files/checkpoints/$step" "$latest/motion_eval"
    if ! git diff --cached --quiet; then
      git commit -m "Record multi6 expert $motion evaluation"
    fi
    git push origin "HEAD:$branch" || { echo "Push failed; retrying $motion" >&2; continue; }
    echo "$motion" >> "$published"
    echo "Published $motion"
  done < "$root/queue.log"
  [[ ${1:-} != --once ]] || break
  sleep 30
done
