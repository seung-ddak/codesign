"""TT 형상 qpos 재생 영상 (code/viz/render_three.Panel 과 같은 카메라·조명, 50 Hz qpos → 25 fps). 실행 전 MUJOCO_GL=egl 또는 osmesa."""
import os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code/viz')); sys.path.insert(0, str(ROOT / 'code/motion_tracking'))


def render_tt(qpos, E, path, title='', sub='TT shape: shared policy tracking', fell=False):
    import imageio, render_three as r3
    from pd_replay import build_model
    rm = r3.rm; p = r3.Panel(build_model('00000011', uniforms=list(E), flat_ground=True, torque_limit=33.5))
    q = np.asarray(qpos).copy(); q[:, :2] -= q[0, :2]; step = max(1, 50 // r3.FPS)
    with imageio.get_writer(str(path), fps=r3.FPS, codec='libx264', quality=7, macro_block_size=8) as w:
        for t in range(0, len(q), step):
            img = p.frame(q[t], q[:t + 1, :2], q[t, :2])
            last = t + step >= len(q)
            st = f"FELL at {t * .02:.1f} s" if (fell and last) else f"t = {t * .02:4.1f} s   moved {np.linalg.norm(q[t, :2]):.2f} m"
            w.append_data(np.asarray(rm.label(img, title, sub, st, bool(fell and last))))
    return path


def render_overlay(q_base, q_test, E, path, title='', sub='', status_fn=None, color=(1.0, 0.55, 0.1, 0.45)):
    """두 rollout 을 한 장면에: q_base = 원래 색 (불투명), q_test = 반투명 색 (color). 같은 시작 위치 기준, 카메라는 q_base 를 따라감.
    status_fn(t) → 아래쪽 상태 문자열. 50 Hz qpos → 25 fps."""
    import imageio, mujoco, render_three as r3
    from pd_replay import build_model
    rm = r3.rm; m = build_model('00000011', uniforms=list(E), flat_ground=True, torque_limit=33.5); p = r3.Panel(m); d2 = mujoco.MjData(m)
    qa, qb = np.asarray(q_base).copy(), np.asarray(q_test).copy(); off = qa[0, :2].copy(); qa[:, :2] -= off; qb[:, :2] -= off
    n = max(len(qa), len(qb)); step = max(1, 50 // r3.FPS)
    with imageio.get_writer(str(path), fps=r3.FPS, codec='libx264', quality=7, macro_block_size=8) as w:
        for t in range(0, n, step):
            a, b = qa[min(t, len(qa) - 1)], qb[min(t, len(qb) - 1)]
            p.d.qpos[:] = a; mujoco.mj_forward(p.m, p.d); p.cam.lookat[:] = [a[0], a[1], 0.2]
            p.r.update_scene(p.d, camera=p.cam, scene_option=p.opt)
            d2.qpos[:] = b; mujoco.mj_forward(m, d2); n0 = p.r.scene.ngeom
            mujoco.mjv_addGeoms(m, d2, p.opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, p.r.scene)
            for g in range(n0, p.r.scene.ngeom): p.r.scene.geoms[g].rgba[:] = color
            rm.decorate(p.r.scene, [], qa[:min(t, len(qa) - 1) + 1, :2], np.zeros((0, 2)))
            img = p.r.render().copy()
            w.append_data(np.asarray(rm.label(img, title, sub, status_fn(t) if status_fn else f't = {t * .02:4.1f} s', False)))
    return path
