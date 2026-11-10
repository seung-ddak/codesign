"""평가 결과 (results/<E>/<motion>.npz) 를 영상으로: 왼쪽 A1 레퍼런스 모션 | 오른쪽 TT 형상 PPO 트래킹. code/viz 의 메쉬 스타일.
출력: video/<E>/<motion>.mp4. 실행 전 `source code/env_osmesa.sh`."""
import json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'osmesa')
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'code/viz')); sys.path.insert(0, str(HERE))
import render_three as r3
from shapes import SHAPES, MOTIONS
rm, Panel, W, H, FPS = r3.rm, r3.Panel, r3.W, r3.H, r3.FPS
EN = {'E0': 'E0 base shape', 'E1': 'E1 short legs (-20%)', 'E2': 'E2 long legs (+20%)', 'E3': 'E3 short body (-20%)', 'E4': 'E4 long body (+20%)'}
summary = {(r['shape'], r['motion']): r for r in json.load(open(HERE / 'results/summary.json'))}


def render(K, M):
    z = np.load(HERE / 'results' / K / f'{M}.npz'); r = summary[(K, M)]
    tt = z['qpos'].astype(np.float64); tt = tt - np.r_[tt[0, :2], np.zeros(tt.shape[1] - 2)]
    a1 = z['a1_qpos'].astype(np.float64); a1 = a1 - np.r_[a1[0, :2], np.zeros(a1.shape[1] - 2)]
    am = r3.a1_model()
    if am.nq != a1.shape[1]:
        raise ValueError(f'A1 qpos size {a1.shape[1]} != model {am.nq}')
    p1, p2 = Panel(am), Panel(rm.gtm.make_physics(z['E']))
    dt = float(z['dt']); n = len(tt) - 1; fell = not r['survived']
    title = f"{EN[K]} · {M.replace('__r00', '')}"
    sub_tt = f"TT PPO tracking  slip {r['slip_m_s']:.3f} m/s (ref {r['ref_slip_m_s']:.3f})  joint {r['joint_rmse_deg']:.1f} deg  root {r['root_rmse_cm']:.1f} cm"
    stride = max(1, round(1 / dt / FPS))

    def frames():
        for t in range(0, len(a1), stride):
            ka = min(t, len(a1) - 1); kt = min(t, n)
            a = rm.label(p1.frame(a1[ka], a1[:ka + 1, :2], a1[ka, :2]), title, 'A1 reference motion',
                         f't = {t * dt:4.1f} s', False)
            st = (f'FELL at {n * dt:.1f} s', True) if fell and t >= n else (f't = {t * dt:4.1f} s   moved {np.linalg.norm(tt[kt, :2]):.2f} m', False)
            b = rm.label(p2.frame(tt[kt], tt[:kt + 1, :2], tt[kt, :2]), title, sub_tt, *st)
            yield np.concatenate([a, b], 1)

    out = HERE / 'video' / K / f'{M}.mp4'; out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): out.unlink()
    try:
        rm.encode_frames(out, frames(), width=2 * W, height=H, fps=FPS)
    finally:
        p1.r.close(); p2.r.close()
    return out


if __name__ == '__main__':
    for K in SHAPES:
        for M in MOTIONS:
            if (K, M) in summary:
                print(render(K, M), flush=True)
