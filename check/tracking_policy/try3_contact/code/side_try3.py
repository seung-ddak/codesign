"""A1 (왼쪽) · try3 접촉 설정 A (가운데) · try3 접촉 설정 F (타원 원뿔, impratio 100, 반복 50/20, 오른쪽) 를 나란히. 정책 재학습 없음.
사용: side_try3.py [모션, 기본 calm_06117] [설정 첫 글자, 기본 F]"""
import json, sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
M = sys.argv[1] if len(sys.argv) > 1 else 'calm_06117'; T = sys.argv[2] if len(sys.argv) > 2 else 'F'
sys.argv = [sys.argv[0], M]
import numpy as np
import run as R
se = R.se; cfg = {n[0]: (n, c) for n, c in se.CONFIGS.items()}
IMP = cfg[T][1]['impratio']
ra, qa = se.run_ck(R.CK, R.E, R.REF, M, cfg['A'][1], multi=True, keep=True); ra.update(shape='E0', config=cfg['A'][0])
rt, qt = se.run_ck(R.CK, R.E, R.REF, M, cfg[T][1], multi=True, keep=True); rt.update(shape='E0', config=cfg[T][0])
json.dump([ra, rt], open(HERE / 'results' / f'{M}_try3_A_vs_{T}.json', 'w'), indent=1); print(ra); print(rt)
z = np.load(R.REF / f'{M}.npz'); out = HERE / 'videos' / f'{M}_A1_try3A_try3{T}_side.mp4'
R.render(qa, qt, z['a1_qpos'], float(z['dt']), out, ra, rt, la='TT try3 (middle): contact A, impratio 1 (training)',
         ld=f'TT try3 (right): elliptic, impratio {IMP}, iter 50', head=f'E0 · {M}   A1 (left) · try3 contact A (middle) · try3 impratio {IMP} (right)',
         short=('A', f'imp{IMP}'), side=True)
print(out)
