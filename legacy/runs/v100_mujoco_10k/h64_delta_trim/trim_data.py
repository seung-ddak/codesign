"""넘어진 에피소드의 마지막 50 step (1초) 을 valid 에서 제거한 사본. 다른 배열·순서는 그대로."""
import numpy as np, glob, json, os, sys
src, dst, TRIM = sys.argv[1], sys.argv[2], 50
stats = dict(files=0, fell=0, steps_before=0, steps_after=0, short_lt2=0)
for f in sorted(glob.glob(f"{src}/e*/variant_*.npz")):
    d = dict(np.load(f))
    v = d["valid"].copy()
    stats["steps_before"] += int(v.sum())
    for b in np.flatnonzero(d["fell"]):
        L = int(v[b].sum())
        newL = max(L - TRIM, 2)  # 학습 로더는 2 step 이상 필요: 에피소드 순서·분할 유지
        stats["short_lt2"] += int(L - TRIM < 2)
        v[b, newL:] = False
        stats["fell"] += 1
    d["valid"] = v
    stats["steps_after"] += int(v.sum())
    out = f.replace(src, dst)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **d)
    stats["files"] += 1
json.dump(stats, open(f"{dst}/trim_stats.json", "w"), indent=1)
print(stats)
