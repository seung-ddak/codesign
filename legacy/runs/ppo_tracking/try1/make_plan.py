"""PPO 트래킹 try1 계획: TT 형상 16 (CALM 파일럿과 같은 선택) x 학습 모션 2,000 (형상당 125) + 평가 모션 192 (형상당 12, 학습 제외)."""
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
base = json.load(open(ROOT / 'runs/shared_calm_v100_48h/plan.json'))
pilot = json.load(open(ROOT / 'runs/calm_tt_policy_pilot/try1/config.json'))
names = list(pilot['shapes'])
assert len(names) == 16
byname = {r['design']: r for r in base['designs']}
designs = [byname[n] for n in names]
ids = [m['id'] for m in base['motions']]
rng = random.Random(20261110); rng.shuffle(ids)
train, held = ids[:2000], ids[2000:2192]
assign = {n: sorted(train[i::16]) for i, n in enumerate(names)}
evals = {n: sorted(held[i::16]) for i, n in enumerate(names)}
c = dict(base['config'])
c.update(num_envs=32, batch_size=32, unroll_length=24, num_minibatches=4, num_updates_per_batch=8,
         learning_rate=1e-3, entropy_cost=0.01, hidden=[256, 128, 64], resident_designs=16,
         train_transitions=15_000_000, seed=0, control_hz=30, physics_hz=240)
keep = set(train) | set(held)
plan = dict(version=1, config=c, topology=base['topology'], source=base['source'],
            motions=[m for m in base['motions'] if m['id'] in keep], designs=designs, eval_designs=[],
            assignments=assign, eval_assignments=evals, clips_per_design=125)
json.dump(plan, open(OUT / 'plan.json', 'w'), indent=1)
print('designs', names); print('train', len(train), 'eval', len(held))
