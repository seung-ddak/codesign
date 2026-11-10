import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/motion_tracking'))
from run_calm_dynamics import retarget
out = Path(__file__).resolve().parent
plan = json.load(open(out / 'plan.json')); name = sys.argv[1]
row = [r for r in plan['designs'] if r['design'] == name][0]
retarget(plan, dict(row, common=[], extra=plan['assignments'][name] + plan['eval_assignments'].get(name, [])), out)
(out / 'ready').mkdir(exist_ok=True); (out / 'ready' / f'{name}.json').write_text('{}')
print('ready', name)
