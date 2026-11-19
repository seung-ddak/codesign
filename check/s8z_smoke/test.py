import sys, json; sys.path.insert(0, '/home/cgr-ugrad-2026/quad_ws/codesign/chain/code')
import s8z_collect as s
from pathlib import Path
import numpy as np
s.OUT = Path('/home/cgr-ugrad-2026/quad_ws/codesign/check/s8z_smoke/out'); s.CHUNK = 2
R = s.S5H / 'references'; sh = ['G_l-20_b-20', 'G_l+20_b+20', 'G_l+00_b+00']
have = set.intersection(*[{f.stem for f in (R / x / '00000011').glob('calm_*.npz')} for x in sh])
g = json.load(open(s.RUNS / 's7_eval/selected.json')); G = sorted(g)[0]
clips = sorted(have)[:2]
p = dict(shapes={x: dict(E=s.token(int(x[3:6]), int(x[8:11])), seen=True, ref_dir=str(R / x / '00000011')) for x in sh},
         motions={x: {G: clips} for x in sh})
import policy_io
_lr = policy_io.load_run
def lr(ck):
    cfg, pol = _lr(ck); cfg.eval_env.ctrl_dt = 0.01; cfg.eval_env.sim_dt = 0.005; return cfg, pol  # 30 Hz 정책을 100 Hz 참조로: 동작 확인용
policy_io.load_run = lr
s.rollout_set(G, g[G], p)
for f in sorted((s.OUT / 'rollouts' / G).glob('*.npz')):
    z = np.load(f); print(f.name, z['valid'].shape, z['valid'].sum(1), z['fell'], z['shape'], z['motion'])
