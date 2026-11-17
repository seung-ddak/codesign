"""calm 모션을 A1 (왼쪽, 원본) 과 TT E0 (오른쪽, try3 PPO 공유 정책, 학습 접촉 설정 A) 에 각각 칸을 나눠 녹화.
legacy/runs/calm_dynamics_1000x100/tt_try1/videos/ep011_calm_06117.mp4 와 같은 2 칸 형식. 사용: two_panel.py [모션, 기본 calm_00026]"""
import os, sys
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
from run import se, CK, REF, E, M, HERE, ROOT
sys.path.insert(0, str(ROOT / 'legacy/runs/v100_mujoco_10k/mesh_render'))
import render_three as r3
rm = r3.rm
cfg = {n[0]: c for n, c in se.CONFIGS.items()}
r, q = se.run_ck(CK, E, REF, M, cfg['A'], multi=True, keep=True)
z = np.load(REF / f'{M}.npz'); dt = float(z['dt']); a1 = np.asarray(z['a1_qpos'], np.float64).copy(); a1[:, :2] -= a1[0, :2]
q = np.asarray(q, np.float64).copy(); q[:, :2] -= q[0, :2]
pa, pt = r3.Panel(r3.a1_model()), r3.Panel(rm.gtm.make_physics(np.asarray(E, np.float64)))
fell = not r['survived']; T = max(len(a1), len(q)) * dt; title = f'E0 · {M}'


def frames():
    for t in np.arange(0, T, 1 / r3.FPS):
        ka = min(int(round(t / dt)), len(a1) - 1); kt = min(int(round(t / dt)), len(q) - 1)
        a = pa.frame(a1[ka], a1[:ka + 1, :2], a1[ka, :2]); b = pt.frame(q[kt], q[:kt + 1, :2], q[kt, :2])
        sa = (f't = {t:4.1f} s   moved {np.linalg.norm(a1[ka, :2]):.2f} m', False)
        st = (f'FELL at {len(q) * dt:.1f} s', True) if fell and kt == len(q) - 1 else (f't = {t:4.1f} s   moved {np.linalg.norm(q[kt, :2]):.2f} m', False)
        yield np.concatenate([rm.label(a, title, 'A1: original CALM motion', *sa),
                              rm.label(b, title, f"TT shape: PPO motion tracking  (joint {r['joint_rmse_deg']:.1f} deg, root {r['root_rmse_cm']:.0f} cm)", *st)], 1)


out = HERE / 'videos' / f'{M}_A1_TT_2panel.mp4'
try: rm.encode_frames(out, frames(), width=2 * r3.W, height=r3.H, fps=r3.FPS)
finally: pa.r.close(); pt.r.close()
print(out, r)
