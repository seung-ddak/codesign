"""s8 rollout 영상: 묶음마다 (가운데 형상 레퍼런스, 다리 짧은/긴 형상 CALM 1 개씩, 넘어진 에피소드 1 개)."""
import numpy as np
from common import RUNS, log
from render_tt import render_tt
S8 = RUNS / 's8_collect'; V = S8 / 'videos'; rng = np.random.default_rng(0)
for g in ['trotForwardSlow', 'trotBackwardSlow', 'trotLeftSlow', 'trotRightSlow', 'trotCWSlow', 'trotCCWSlow']:
    picks = []
    for s, ref in [('G_l+00_b+10', True), ('G_l-20_b-10', False), ('G_l+20_b+10', False)]:
        d = np.load(S8 / s / f'{g}__r00.npz'); m = d['motions']
        ok = [i for i in range(len(m)) if not d['fell'][i] and ((m[i] == f'{g}__r00') == ref)]
        picks.append((s, d, int(rng.choice(ok)), 'ref' if ref else 'calm'))
    for s in sorted(p.name for p in S8.glob('G_*')):
        d = np.load(S8 / s / f'{g}__r00.npz')
        if d['fell'].any(): picks.append((s, d, int(np.flatnonzero(d['fell'])[0]), 'fell')); break
    for s, d, i, tag in picks:
        dst = V / g / f'{tag}_{s}_{d["motions"][i]}.mp4'; dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists(): continue
        q = d['qpos'][i][d['valid'][i]]
        render_tt(q, d['E'], dst, fell=bool(d['fell'][i]), title=f's8 {g} · {s} · {d["motions"][i]} · {"넘어짐" if d["fell"][i] else "생존"}',
                  sub='학습에 쓰지 않은 형상 rollout')
        log(f's8 영상 {dst.name}')
