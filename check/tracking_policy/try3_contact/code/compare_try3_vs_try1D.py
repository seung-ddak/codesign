"""A1 vs try3 (발 미끄러짐 감점 -2.5, 접촉 설정 A = 학습 설정) vs try1 (감점 없음) 을 접촉 설정 D (타원 원뿔, impratio 10, 반복 50/20) 로 실행.
E0, CALM 모션 하나. 영상: A1 (왼쪽) + TT try3-A (원래 색) · try1-D (반투명 주황) 겹침. 사용: compare_try3_vs_try1D.py [모션, 기본 calm_06117]"""
import json, sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관
sys.path.insert(0, str(CODE_DIR))
M = sys.argv[1] if len(sys.argv) > 1 else 'calm_06117'
sys.argv = [sys.argv[0], M]
import numpy as np
import run as R
se = R.se; P = ROOT / 'legacy/runs/ppo_tracking'
CK1 = P / 'try1/stages/000/files/checkpoints/6881280'; CK3 = P / 'try3/stages/000/files/checkpoints/100270080'
cfg = {n[0]: c for n, c in se.CONFIGS.items()}
REFD = HERE / 'references/E0/00000011'  # E0 로 리타겟 (try3 의 30 Hz A1 레퍼런스, 토크 33.5)
r3, q3 = se.run_ck(CK3, R.E, REFD, M, cfg['A'], multi=True, keep=True); r3.update(shape='E0', config='try3, contact A')
r1, q1 = se.run_ck(CK1, R.E, REFD, M, cfg['D'], multi=True, keep=True); r1.update(shape='E0', config='try1, contact D')
json.dump([r3, r1], open(HERE / 'results' / f'{M}_try3A_vs_try1D.json', 'w'), indent=1); print(r3); print(r1)
z = np.load(REFD / f'{M}.npz'); out = HERE / 'videos' / f'{M}_A1_try3A_vs_try1D.mp4'
R.render(q3, q1, z['a1_qpos'], float(z['dt']), out, r3, r1,
         la='TT try3: slip penalty -2.5, contact A (pyramidal, impratio 1)', ld='TT try1: no slip penalty, contact D (elliptic, impratio 10, iter 50)',
         head=f'E0 · {M}   A1 vs slip-penalty policy vs higher impratio   (A1 left, TT right)', short=('try3', 'try1-D'))
print(out)
