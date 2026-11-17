"""try1 (발 미끄러짐 감점 없음, 동시 환경 32, 690 만 step) vs try3 (감점 -2.5, 동시 환경 1,024, 1 억 step) 을 E0 에서 같은 CALM 모션으로 비교.
접촉 설정은 둘 다 학습과 같은 A. 영상: A1 (왼쪽) + TT try1 (원래 색) · try3 (반투명 주황) 겹침. 사용: compare_try1_try3.py [모션, 기본 calm_00238]"""
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
A = next(c for n, c in se.CONFIGS.items() if n[0] == 'A')
REFD = HERE / 'references/E0/00000011'  # E0 로 새로 리타겟 (try3 의 30 Hz A1 레퍼런스에서, 토크 33.5). 두 정책 모두 E0 에서 이 모션을 학습하지 않음
r1, q1 = se.run_ck(CK1, R.E, REFD, M, A, multi=True, keep=True); r1.update(shape='E0', config='try1')
r3, q3 = se.run_ck(CK3, R.E, REFD, M, A, multi=True, keep=True); r3.update(shape='E0', config='try3')
json.dump([r1, r3], open(HERE / 'results' / f'{M}_try1_vs_try3.json', 'w'), indent=1); print(r1); print(r3)
z = np.load(REFD / f'{M}.npz')
out = HERE / 'videos' / f'{M}_A1_try1_vs_try3.mp4'
R.render(q1, q3, z['a1_qpos'], float(z['dt']), out, r1, r3, la='TT try1: no slip penalty, 6.9M steps', ld='TT try3: slip penalty -2.5, 100M steps',
         head=f'E0 · {M} · contact A (training)   (same scene: A1 left, TT right)', short=('try1', 'try3'))
print(out)
