"""PPO 트래킹 try2: 형상 20 (try1 의 16 + E 공간에서 가장 먼 4) × 모션 10,000 (모션당 형상 1, 형상당 500). 발 미끄러짐 감점 -10."""
import json, random
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; OUT = Path(__file__).resolve().parent
base = json.load(open(ROOT / 'runs/shared_calm_v100_48h/plan.json'))
t1 = json.load(open(ROOT / 'runs/ppo_tracking/try1/plan.json'))
names = [r['design'] for r in t1['designs']]
pool = {r['design']: np.array(r['E']) for r in base['designs']}
while len(names) < 20:  # farthest-point: 이미 고른 형상에서 E 거리가 가장 먼 형상
    rest = [n for n in pool if n not in names]
    names.append(max(rest, key=lambda n: min(np.linalg.norm(pool[n] - pool[m]) for m in names)))
byname = {r['design']: r for r in base['designs']}
ids = [m['id'] for m in base['motions']]
rng = random.Random(20261110); rng.shuffle(ids)
assign = {n: sorted(ids[i::20]) for i, n in enumerate(names)}
c = dict(t1['config']); c.update(resident_designs=20, train_transitions=2_000_000_000, penalty_foot_slip=-10.0, seed=0)
plan = dict(version=1, config=c, topology=base['topology'], source=base['source'], motions=base['motions'],
            designs=[byname[n] for n in names], eval_designs=[], assignments=assign, eval_assignments={n: [] for n in names}, clips_per_design=500)
json.dump(plan, open(OUT / 'plan.json', 'w'), indent=1)
print(names); print({n: len(v) for n, v in list(assign.items())[:3]}, sum(len(v) for v in assign.values()))
