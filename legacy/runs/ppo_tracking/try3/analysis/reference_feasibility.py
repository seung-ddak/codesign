"""리타겟 레퍼런스의 물리적 가능성 점검 (형상 20 × 모션 500 = 10,000). GPU 불필요.
지표 (리타겟 때 계산·저장된 값): 필요 토크 (역동역학 p99, 최대), 외력이 필요한 프레임 비율 (inv_bad_frac),
발 위치 오차, 발 미끄러짐, 잔여 몸통 힘 p95 (몸무게 대비 %)."""
import json
from pathlib import Path
import numpy as np
O = Path(__file__).resolve().parents[1]
plan = json.load(open(O / 'plan.json'))
rows = []
for d in plan['designs']:
    name = d['design']
    for mid in plan['assignments'][name]:
        f = O / f'references/{name}/00000011/calm_{mid:05d}.npz'
        with np.load(f) as z:
            rows.append(dict(design=name, motion=mid, tau_p99=float(z['inv_tau_p99']), tau_max=float(z['inv_tau_max']),
                             bad_frac=float(z['inv_bad_frac']), res_force_p95_pct=float(z['inv_res_force_p95_pct']),
                             foot_err_mm=float(z['fix_track_err']) * 1000, slip=float(z['fix_slip']), dq_max_deg=float(z['fix_dq_max_deg'])))
json.dump(rows, open(O / 'analysis/reference_feasibility_rows.json', 'w'))
A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ('design', 'motion')}
q = lambda x: [float(np.percentile(x, p)) for p in (50, 90, 99, 100)]
TAU = 33.5
summary = dict(n=len(rows), percentiles='50/90/99/max', **{k: q(v) for k, v in A.items()},
               frac_tau_max_over_limit=float((A['tau_max'] > TAU).mean()), frac_tau_p99_over_limit=float((A['tau_p99'] > TAU).mean()),
               frac_bad_over_5pct=float((A['bad_frac'] > .05).mean()), frac_bad_over_10pct=float((A['bad_frac'] > .10).mean()),
               frac_foot_err_over_10mm=float((A['foot_err_mm'] > 10).mean()))
feasible = (A['tau_p99'] <= TAU) & (A['bad_frac'] <= .05) & (A['foot_err_mm'] <= 10)
summary['frac_feasible_tau_p99_bad5_foot10mm'] = float(feasible.mean())
by = {}
for r, ok in zip(rows, feasible):
    by.setdefault(r['design'], []).append(ok)
summary['feasible_by_design'] = {k: float(np.mean(v)) for k, v in by.items()}
json.dump(summary, open(O / 'analysis/reference_feasibility.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != 'feasible_by_design'}, indent=1)); print({k: round(v, 3) for k, v in summary['feasible_by_design'].items()})
