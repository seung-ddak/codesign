"""5 단계: 학습 형상 격자 리타겟.
형상 = 다리 변화 legs × 몸통 변화 bodies (1 단계 결정). 형상마다 레퍼런스 25 개 전부 + CALM 모션 (4 단계 selected.json) 은 모션 하나당 형상 2 개
(레퍼런스 묶음 안에서 형상 쌍을 돌려 가며 배정해 묶음 × 형상 칸이 고르게 채워지도록). CALM (30 Hz) 은 그대로, 레퍼런스 25 개 (50 Hz) 는 30 Hz 로 다시 표본 추출 후 리타겟 (2026-11-12 사용자 결정: 학습 30 Hz).
출력: runs/s5_retarget/ (a1_reference/, references/<형상>/00000011/, ready/<형상>.json, assignment.json, report.html)."""
import itertools, json, os
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from common import ROOT, RUNS, log, report, table
from shapes_grid import token, name, dims
OUT = RUNS / 's5_retarget'; A1 = OUT / 'a1_reference'
REF = ROOT / 'legacy/artifacts/randomized_reference'; HZ = 30


def convert(src):
    from plan_calm_dynamics import prepare_reference
    dst = A1 / f"calm_{Path(src).parent.parent.name}_{Path(src).stem}.npz"
    if not dst.exists(): prepare_reference(Path(src), dst, hz=HZ)
    return dst.stem


def retarget(job):
    os.environ['OMP_NUM_THREADS'] = '1'
    import pd_replay as pd
    shape, E, clips = job; dst = OUT / 'references' / shape
    todo = [c for c in clips if not (dst / '00000011' / f'{c}.npz').exists()]
    if todo: pd.retarget_tt(str(A1), str(dst), only=todo, choices=['00000011'], uniforms=E, torque_limit=33.5)
    return shape, len(todo)


def assign(selected, shapes, per_motion=2):
    """묶음마다 형상 쌍을 순환 배정. 반환 {형상: [clip]}, {묶음: {형상: [clip]}}."""
    pairs = list(itertools.combinations(shapes, per_motion)); by_shape = {s: [] for s in shapes}; by_group = {}
    for g, (ref, clips) in enumerate(sorted(selected.items())):
        by_group[ref] = {s: [] for s in shapes}
        for i, c in enumerate(clips):
            for s in pairs[(i + g) % len(pairs)]: by_shape[s].append(c); by_group[ref][s].append(c)
    return by_shape, by_group


def main(legs, bodies, workers=14):
    OUT.mkdir(parents=True, exist_ok=True); A1.mkdir(exist_ok=True)
    shapes = {name(l, b): token(l, b) for l in legs for b in bodies}
    sel = json.load(open(RUNS / 's4_calm/selected.json'))
    with Pool(workers) as p:
        names = {}
        for ref, files in sel.items():
            names[ref] = sorted(p.map(convert, files, chunksize=20))
    refs = sorted(f.stem for f in REF.glob('*.npz'))
    for r in refs:
        if not (A1 / f'{r}.npz').exists():
            from plan_calm_dynamics import resample_reference
            resample_reference(REF / f'{r}.npz', A1 / f'{r}.npz', hz=HZ)
    by_shape, by_group = assign(names, list(shapes))
    json.dump(dict(shapes={k: dict(E=v) for k, v in shapes.items()}, references=refs, by_group=by_group), open(OUT / 'assignment.json', 'w'), indent=1)
    jobs = [(s, shapes[s], (refs + by_shape[s])[c::3]) for s in shapes for c in range(3)]
    with Pool(workers) as p:
        for s, n in p.imap_unordered(retarget, jobs): log(f's5 retarget {s} +{n}')
    (OUT / 'ready').mkdir(exist_ok=True); rows = []
    for s in shapes:
        want = set(refs + by_shape[s]); have = {f.stem for f in (OUT / 'references' / s / '00000011').glob('*.npz')}
        miss = sorted(want - have)
        if not miss: (OUT / 'ready' / f'{s}.json').write_text(json.dumps(dict(clips=len(have))))
        import csv
        summ = list(csv.DictReader(open(next((OUT / 'references' / s).glob('summary*.csv'))))) if list((OUT / 'references' / s).glob('summary*.csv')) else []
        bad = np.mean([float(r['inv_bad_frac']) > .05 for r in summ]) if summ else float('nan')
        rows.append((s, dims(*map(float, s[3:].replace('_b', ' ').split())), len(refs), len(by_shape[s]), len(have), len(miss), f'{bad * 100:.0f} %'))
    report(OUT / 'report.html', '5 단계: 학습 형상 리타겟', [
        ('형상별 리타겟', table(['형상', '치수', '레퍼런스', 'CALM 모션', '완료', '누락', '역동역학 무리 클립 (>5 % 프레임)'], rows, lambda i: 'bad' if rows[i][5] else '')),
        ('묶음 × 형상 CALM 모션 수', table(['묶음'] + list(shapes), [[g.replace('__r00', '')] + [len(v[s]) for s in shapes] for g, v in by_group.items()]))],
        [('형상', f'다리 {legs} % × 몸통 {bodies} % (shapes_grid.token)'), ('배정', '레퍼런스 25 개 = 모든 형상, CALM 모션 = 하나당 형상 2 개 (묶음 안에서 형상 쌍 순환)'),
         ('리타겟', 'pd_replay.retarget_tt, choice 00000011, 토크 33.5, 30 Hz')])
    return dict(shapes=shapes, missing=sum(r[5] for r in rows))
