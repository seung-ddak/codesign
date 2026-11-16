#!/bin/bash
# 커밋 안 된 파일을 폴더 순서대로 약 300 MB 씩 나눠 커밋·푸시
cd /home/cgr-ugrad-2026/quad_ws/codesign
git status --porcelain --untracked-files=all | sed 's/^...//' | sort > /tmp/claude-1000/pending.txt
n=0; size=0; batch=()
flush() {
  [ ${#batch[@]} -eq 0 ] && return
  n=$((n+1)); printf '%s\0' "${batch[@]}" | xargs -0 git add -A --
  git commit -qm "업로드 $n: ${batch[0]%/*} 외 ${#batch[@]} 파일

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin experiment/h64-delta-trim && echo "OK $n ${#batch[@]} files $((size/1000000)) MB" || { echo "FAIL $n"; exit 1; }
  batch=(); size=0
}
while IFS= read -r f; do
  s=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ $((size+s)) -gt 300000000 ]; then flush; fi
  batch+=("$f"); size=$((size+s))
done < /tmp/claude-1000/pending.txt
flush; echo DONE
