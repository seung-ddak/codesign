"""s5 리타겟 (학습 형상 9 개 × CALM 모션) 의 역동역학 다리 토크가 한계 33.5 N·m 를 넘는 모션 수 집계."""
import glob, json, os, numpy as np
R = '/home/cgr-ugrad-2026/quad_ws/codesign/chain/runs/s5_retarget'; LIM = 33.5
a = json.load(open(f'{R}/assignment.json')); grp = {}
for g, per in a['by_group'].items():
    for s, ms in per.items():
        for m in ms: grp[(s, m)] = g
rows = []
for s in a['shapes']:
    for f in sorted(glob.glob(f'{R}/references/{s}/00000011/*.npz')):
        m = os.path.basename(f)[:-4]
        if (s, m) not in grp: continue
        z = np.load(f); tau = np.abs(z['inv_tau'][:, :12]); over = (tau > LIM).any(1)
        rows.append(dict(shape=s, motion=m, group=grp[(s, m)], tau_max=float(tau.max()), tau_p99=float(np.percentile(tau, 99)),
                         over_frac=float(over.mean()), over_frames=int(over.sum()), frames=len(tau)))
json.dump(rows, open('rows.json', 'w'))
print(len(rows), 'pairs', sum(r['over_frames'] > 0 for r in rows), 'exceed', len(grp), 'assigned')
