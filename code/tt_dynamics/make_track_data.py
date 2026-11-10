"""Copy of the 10k PD data keeping, per episode, the longest run of well-tracked 1 s windows.

The kept run is moved to the start of the row (loader reads valid[:L] from step 0);
start_frame is shifted to match. Rows, files and order are unchanged, so the
train/val split is identical. Fell episodes lose their last 1 s first (as h64_delta_trim).
"""
import glob, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from track_filter import window_errors

src, dst = sys.argv[1], sys.argv[2]
YAW, POS, PER, TRIM = float(sys.argv[3]), float(sys.argv[4]), 50, 50
stats = dict(yaw_thresh=YAW, pos_thresh=POS, files=0, episodes=0, steps_before=0, steps_after=0,
             episodes_ge64=0, episodes_lt2=0, shifted=0)
for f in sorted(glob.glob(f"{src}/e*/variant_*.npz")):
    d = dict(np.load(f))
    v = d["valid"].copy()
    for b in range(len(v)):
        L = int(v[b].sum())
        stats["steps_before"] += L
        if d["fell"][b]:
            L = max(L - TRIM, 0)
        ye, pe = window_errors(d["qpos"][b], L, int(str(d["motions"][b]).split("_")[1]), int(d["start_frame"][b]))
        ok = (ye < YAW) & (pe < POS)
        best, run, best_s, s = 0, 0, 0, 0
        for k, g in enumerate(ok):
            if g:
                if run == 0: s = k
                run += 1
                if run > best: best, best_s = run, s
            else:
                run = 0
        a, n = best_s * PER, best * PER
        if n < 2:                      # keep 2 steps so row order / split stay fixed (as trim_data.py)
            a, n = 0, 2
            stats["episodes_lt2"] += 1
        if a:
            for k in ("qpos", "qvel", "contact"):
                d[k][b, :n + 1] = d[k][b, a:a + n + 1]
            for k in ("action", "target_q"):
                d[k][b, :n] = d[k][b, a:a + n]
            d["start_frame"][b] = d["start_frame"][b] + int(round(a * 30 / 50))
            stats["shifted"] += 1
        v[b] = False
        v[b, :n] = True
        stats["episodes"] += 1
        stats["steps_after"] += n
        stats["episodes_ge64"] += int(n >= 64)
    d["valid"] = v
    out = f.replace(src, dst)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **d)
    stats["files"] += 1
json.dump(stats, open(f"{dst}/track_stats.json", "w"), indent=1)
print(stats)
