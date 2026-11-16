"""CALM 회전 문제 진단: 고정 조이스틱 명령 (10 초, 처음 1 초는 0 에서 서서히) 으로 collect_calm_joystick 을 그대로 실행.
품질 검사·중복 검사는 끄고 (정지 명령도 저장), 명령 함수만 바꾼다. 실행: rlgpu python. 출력: diag/accepted/*.npz, diag/plan.json"""
import json, os, sys
from pathlib import Path
import numpy as np
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'chain/code'))
import collect_calm_joystick as cc
FPS = cc.FPS; SEC = 10
CASES = [('stop', 0, 0, 0), ('stop_b', 0, 0, 0), ('fwd0.4', .4, 0, 0), ('right0.2', 0, -.2, 0), ('left0.2', 0, .2, 0),
         ('wz-0.5', 0, 0, -.5), ('wz-1.0', 0, 0, -1.), ('wz-1.5', 0, 0, -1.5), ('wz+1.0', 0, 0, 1.), ('fwd0.4_wz-0.5', .4, 0, -.5), ('wz-0.05', 0, 0, -.05), ('wz-0.2', 0, 0, -.2)]
OUT = HERE / os.environ.get('DIAG_OUT', 'diag'); OUT.mkdir(exist_ok=True)


def cmd_of(plan):
    n = plan['frames']; c = np.tile(np.array(plan['const'], float), (n, 1)); r = FPS
    s = np.linspace(0, 1, r); c[:r] *= (s * s * (3 - 2 * s))[:, None]; return c


if __name__ == '__main__':
    plans = [dict(id=i, seed=10_000 + i, frames=SEC * FPS, seconds=SEC, initial_pose='standing', joystick=True, refs=[], weights=None,
                  segments=[dict(index=0, start=0, end=SEC * FPS, label=name)], signature=f'diag{i}', const=[vx, vy, wz])
             for i, (name, vx, vy, wz) in enumerate(CASES)]
    (OUT / 'plan.json').write_text(json.dumps(dict(motions=plans)))
    cc.commands_for = cmd_of; cc.quality = lambda plan, root, q: (True, 'pass', [])
    cc.resegment = lambda p: p
    sys.argv = [sys.argv[0], '--out', str(OUT), '--batch', '12', '--max-attempts', '12', '--duplicate-distance', '1e-6',
                '--rl-games-wheel', str(ROOT / 'legacy/runs/menagerie_calm_check/try4/runtime/rl_games-1.1.4-py3-none-any.whl')]
    cc.main()
