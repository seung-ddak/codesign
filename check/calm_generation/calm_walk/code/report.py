"""calm_walk 결과 보고서 (rows.json + 접촉 패턴)."""
import glob, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관; sys.path.insert(0, str(ROOT / 'chain/code'))
from common import report, table
r = json.load(open(HERE / 'rows.json')); rows, bins = [], []
for g in ['walk', 'trot']:
    x = [a for a in r if a['gait'] == g]; o = np.array([a['over_frac'] for a in x]); s = np.array([a['speed'] for a in x]); m = {}
    for a in x: m.setdefault(a['motion'], []).append(a['over_frac'])
    c = np.concatenate([np.load(f)['fix_contact'] for f in glob.glob(str(HERE / g / 'a1/*.npz'))]); n = np.bincount(c.sum(1), minlength=5) / len(c)
    rows.append((f'{g} Slow 명령', len(m), len(x), f'{(o > 0).mean() * 100:.0f} %', f'{(o > .01).mean() * 100:.0f} %', f"{sum(all(v == 0 for v in vs) for vs in m.values())}/{len(m)}",
                 f'{np.median(s):.2f}', f"{np.median([a['tau_max'] for a in x]):.1f}", ' / '.join(f'{v * 100:.0f}' for v in n[2:])))
    for a, b in [(0, .2), (.2, .4), (.4, 9)]:
        k = (s >= a) & (s < b); bins.append((g, f'{a}~{b if b < 9 else ""} m/s', int(k.sum()), f'{(o[k] > 0).mean() * 100:.0f} %' if k.any() else '-'))
report(HERE / 'report.html', 'CALM walk 명령 토크 확인', [
    ('요약', table(['명령', '모션', '쌍 (×형상 9)', '초과 있는 쌍', '초과 프레임 >1 % 쌍', '형상 9 개 모두 초과 없는 모션', '평균 속력 중앙 (m/s)', '최대 토크 중앙 (N·m)', '접촉 발 2/3/4 개 비율 %'], rows)),
    ('속력 구간별 초과', table(['명령', '속력', '쌍', '초과 있는 쌍'], bins))],
    [('생성', 'chain/code/collect_calm_joystick.py (4 단계와 같은 CALM HLC·LLC, step 0.5 초 heading), 계획 calm_joystick.make_plan(48, seed 4242), 속도 배율 0.5~2.5'),
     ('명령', 'walk = walk*Slow 6 개 레퍼런스 속도, trot = trot*Slow 6 개. 같은 seed → 같은 구간 구성'),
     ('토크', '형상 9 개 리타겟 역동역학 inv_tau, 한계 33.5 N·m'), ('코드', 'check/calm_walk/code/run.py, report.py')])
