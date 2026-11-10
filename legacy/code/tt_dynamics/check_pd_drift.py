"""모션별 몸통 전진 속도 부호 점검 (remind 8-36). PD/PPO rollout NPZ 와 리타겟팅 레퍼런스를 같은 표에 놓는다.

사용: python check_pd_drift.py <rollout.npz>... [--ref runs/pipeline_v1/retargeting/e000/00000011]
열: body_dx/s = 첫 프레임 heading 기준 전진 속도 (episode 중앙값), z = 몸통 높이 평균. 레퍼런스 열은 fix_base_pos 기준.
"""
import argparse
from pathlib import Path

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("npz", nargs="+", type=Path)
p.add_argument("--ref", type=Path, help="리타겟팅 레퍼런스 폴더 (<motion>.npz 에 fix_base_pos)")
p.add_argument("--min-steps", type=int, default=20)
args = p.parse_args()


def body_dx_per_s(q, L, dt):
    w, x, y, z = q[0, 3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    d = q[L - 1, :2] - q[0, :2]
    return (np.cos(yaw) * d[0] + np.sin(yaw) * d[1]) / ((L - 1) * dt)


tables = {}
motions = None
for f in args.npz:
    d = np.load(f, allow_pickle=True)
    motions, q, valid, clip, dt = list(d["motions"]), d["qpos"], d["valid"].astype(bool), d["clip"], float(d["dt"])
    rows = {}
    for i in range(len(q)):
        L = int(valid[i].sum())
        if L < args.min_steps:
            continue
        c = int(np.bincount(clip[i, :L].astype(int)).argmax())
        rows.setdefault(c, []).append((body_dx_per_s(q[i], L + 1, dt), q[i, :L, 2].mean()))
    tables["/".join(f.parts[-4:-1])] = {c: np.median(np.array(v), 0) for c, v in rows.items()}

ref = {}
if args.ref:
    for c, m in enumerate(motions):
        z = np.load(args.ref / f"{m}.npz", allow_pickle=True)
        bp = z["fix_base_pos"]
        ref[c] = ((bp[-1, 0] - bp[0, 0]) / ((len(bp) - 1) * float(z["dt"])), bp[:, 2].mean())

cols = list(tables) + (["reference"] if ref else [])
print(f"{'motion':24s}" + "".join(f"{c[-26:]:>28s}" for c in cols))
print(f"{'':24s}" + "".join(f"{'dx/s      z':>28s}" for _ in cols))
mismatch = 0
for c, m in enumerate(motions):
    line = f"{m:24s}"
    for name in tables:
        v = tables[name].get(c)
        line += f"{'':>14s}" if v is None else f"{v[0]:+14.3f}{v[1]:14.3f}"
    if ref:
        line += f"{ref[c][0]:+14.3f}{ref[c][1]:14.3f}"
        for name in tables:
            v = tables[name].get(c)
            if v is not None and abs(ref[c][0]) > 0.1 and np.sign(v[0]) != np.sign(ref[c][0]):
                mismatch += 1
    print(line)
if ref:
    print(f"\n레퍼런스 |dx/s|>0.1 모션 중 부호 불일치: {mismatch}")
