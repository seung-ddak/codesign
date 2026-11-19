"""속도→z 배정이 바뀐 대표 클립 2 개 (Backward→Left, Left→Forward; z 1·2 위 차이가 가장 큰 것) A1 영상. three_calm.a1_only 재사용."""
import json, sys
from pathlib import Path
import numpy as np
C = Path('/home/cgr-ugrad-2026/quad_ws/codesign/chain/code'); sys.path.insert(0, str(C)); import os; os.chdir(C)
from three_calm import a1_only
from common import RUNS
OUT = Path(__file__).resolve().parents[1]; a = json.load(open(OUT / 'assign.json')); N = ['Forward', 'Backward', 'Left', 'Right', 'CW', 'CCW']
for s, zt in [[(1, 2), (2, 0)][int(sys.argv[1])]]:
    c = max((c for c in a['clips'] if c['speed'] == s and c['z'] == zt), key=lambda c: sorted(c['mean_cos'])[-1] - sorted(c['mean_cos'])[-2])
    f = Path(c['file']); name = f"calm_{f.parent.parent.name}_{f.stem}"
    from plan_calm_dynamics import prepare_reference
    dst = OUT / 'a1' / f'{name}.npz'; dst.parent.mkdir(exist_ok=True)
    if not dst.exists(): prepare_reference(f, dst, hz=30)
    y = np.load(dst); z = dict(dt=y['dt'], a1_qpos=np.c_[y['fix_base_pos'], y['fix_base_quat'], y['fix_q']])
    a1_only(z, OUT / 'videos' / f'{N[s]}_to_{N[zt]}.mp4', M=name, dirs=f'speed: {N[s]} -> z: {N[zt]}  cos ' + ' '.join(f'{n[:3]} {v:.2f}' for n, v in zip(N, c['mean_cos'])))
    print(name, c['mean_cos'], c['votes'])
