"""짧은 확인 (2026-11-17 사용자 요청): 4 단계 CALM 조이스틱 생성을 walk Slow 레퍼런스 명령으로 → 형상 9 개 리타겟 → 역동역학 토크 초과.
같은 seed·개수의 trot Slow 명령 생성과 비교. CALM 정책은 같고 (걸음새는 CALM 이 정한다) 레퍼런스는 조이스틱 명령 속도만 정한다.
단계: plan → collect (rlgpu, Isaac Gym) → A1 30 Hz 변환 → 형상 9 개 리타겟 (inv_tau) → 집계.
실행: transformer-transformer venv python code/run.py [N]. 출력: <gait>/{plan.json, accepted/, a1/, ret/<형상>/}, rows.json"""
import json, os, subprocess, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관; CHAIN = ROOT / 'chain/code'
sys.path.insert(0, str(CHAIN)); sys.path.insert(0, str(ROOT / 'code/motion_tracking'))
import calm_joystick as cj
from s4_calm import RLGPU, WHEEL
N = int(sys.argv[1]) if len(sys.argv) > 1 else 48; SEED = 4242; LIM = 33.5
REFS = {'walk': [r.replace('trot', 'walk') for r in cj.TROT], 'trot': cj.TROT}
SHAPES = json.load(open(ROOT / 'chain/runs/s5_retarget/assignment.json'))['shapes']


def collect(g):
    d = HERE / g; d.mkdir(exist_ok=True)
    if not (d / 'plan.json').exists(): (d / 'plan.json').write_text(json.dumps(dict(motions=cj.make_plan(N, SEED, refs=REFS[g]))))
    if json.loads((d / 'status.json').read_text()).get('status') == 'complete' if (d / 'status.json').exists() else False: return
    with open(HERE / f'logs/collect_{g}.log', 'a') as fh:
        subprocess.run([f'{RLGPU}/bin/python', '-u', str(CHAIN / 'collect_calm_joystick.py'), '--out', str(d), '--batch', '16', '--max-attempts', str(3 * N),
                        '--rl-games-wheel', str(WHEEL)], stdout=fh, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MPLCONFIGDIR='/tmp/calm-mpl', LD_LIBRARY_PATH=f'{RLGPU}/lib'))


def retarget(job):
    os.environ['OMP_NUM_THREADS'] = '1'; import pd_replay as pd
    g, s = job; a1 = HERE / g / 'a1'; dst = HERE / g / 'ret' / s; clips = sorted(p.stem for p in a1.glob('*.npz'))
    todo = [c for c in clips if not (dst / '00000011' / f'{c}.npz').exists()]
    if todo: pd.retarget_tt(str(a1), str(dst), only=todo, choices=['00000011'], uniforms=SHAPES[s]['E'], torque_limit=33.5)
    return g, s


def main():
    (HERE / 'logs').mkdir(exist_ok=True)
    from plan_calm_dynamics import prepare_reference
    for g in REFS: collect(g)
    for g in REFS:
        a1 = HERE / g / 'a1'; a1.mkdir(exist_ok=True)
        for f in sorted((HERE / g / 'accepted').glob('*.npz')):
            if not (a1 / f'calm_{f.stem}.npz').exists(): prepare_reference(f, a1 / f'calm_{f.stem}.npz', hz=30)
    with Pool(14) as p: list(p.imap_unordered(retarget, [(g, s) for g in REFS for s in SHAPES]))
    rows = []
    for g in REFS:
        plan = {m['id']: m for m in json.load(open(HERE / g / 'plan.json'))['motions']}
        for s in SHAPES:
            for f in sorted((HERE / g / 'ret' / s / '00000011').glob('*.npz')):
                z = np.load(f); tau = np.abs(z['inv_tau'][:, :12]); over = tau > LIM; a = np.load(HERE / g / 'a1' / f.name)
                v = np.linalg.norm(np.diff(a['fix_base_pos'][:, :2], axis=0), axis=1).mean() / float(a['dt'])
                rows.append(dict(gait=g, shape=s, motion=f.stem, speed=float(v), tau_max=float(tau.max()), over_frac=float(over.any(1).mean()),
                                 joint_over=[float(over[:, j::3].any(1).mean()) for j in range(3)], seconds=plan[int(f.stem[5:])]['seconds']))
    json.dump(rows, open(HERE / 'rows.json', 'w'), indent=1)
    for g in REFS:
        r = [x for x in rows if x['gait'] == g]; o = np.array([x['over_frac'] for x in r])
        print(g, len(r), 'pairs  exceed any', f'{(o > 0).mean():.2f}', '>1%', f'{(o > .01).mean():.2f}', 'mean over', f'{o.mean():.3f}',
              'speed', f"{np.median([x['speed'] for x in r]):.2f}", 'tau_max median', f"{np.median([x['tau_max'] for x in r]):.1f}", flush=True)


if __name__ == '__main__':
    main()
