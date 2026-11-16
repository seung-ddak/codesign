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


REAL_C, PRED_C = (0.15, 0.75, 0.35, 1.0), (1.0, 0.5, 0.1, 0.55)  # MuJoCo 실제 = 초록 (불투명), TT 예측 = 주황 (반투명)


def _tint(m, scene, start, rgba):
    import mujoco
    for g in range(start, scene.ngeom):
        s = scene.geoms[g]
        if s.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[s.objid] > 0: s.rgba[:] = rgba


def _pole(scene, x, y, r, h):
    import mujoco
    if scene.ngeom >= scene.maxgeom: return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([r, r, h / 2]),
                        np.array([x, y, h / 2]), np.eye(3).ravel(), np.array([0.85, 0.18, 0.12, 1], np.float32))
    scene.ngeom += 1


def render_task_overlay(q_real, q_pred, E, task, path, title='', fell_t=None):
    """과제 환경 (트랙 경계·슬라럼 폴) 한 장면에 실제 MuJoCo 재생 (초록) 과 TT 예측 (주황 반투명) 을 겹쳐 그림.
    q_* = 50 Hz qpos (s0 기준 world 좌표, 원점 출발). 카메라는 두 몸통 중점을 따라감, 아래에 두 위치 차이 (m)."""
    import imageio, mujoco, render_three as r3
    from pd_replay import build_model
    rm = r3.rm; m = build_model('00000011', uniforms=list(E), flat_ground=True, torque_limit=33.5); p = r3.Panel(m); d2 = mujoco.MjData(m)
    p.cam.distance, p.cam.elevation = 2.6, -32.
    spec = task.describe(); lines = task.guide_lines()
    qa, qb = np.asarray(q_real), np.asarray(q_pred); n = max(len(qa), len(qb)); step = max(1, 50 // r3.FPS)
    with imageio.get_writer(str(path), fps=r3.FPS, codec='libx264', quality=7, macro_block_size=8) as w:
        for t in range(0, n, step):
            ia, ib = min(t, len(qa) - 1), min(t, len(qb) - 1); a, b = qa[ia], qb[ib]
            p.d.qpos[:] = a; mujoco.mj_forward(p.m, p.d)
            p.cam.lookat[:] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.15]
            p.r.update_scene(p.d, camera=p.cam, scene_option=p.opt); sc = p.r.scene
            _tint(m, sc, 0, REAL_C)
            d2.qpos[:] = b; mujoco.mj_forward(m, d2); n0 = sc.ngeom
            mujoco.mjv_addGeoms(m, d2, p.opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, sc)
            _tint(m, sc, n0, PRED_C)
            for pts, solid in lines:
                for u, v in zip(pts[:-1:2], pts[2::2]): rm.add_line(sc, u, v, (1, 1, 1, .9) if solid else (.2, .3, .5, .9), .006 if solid else .012)
            for x, y in spec.get('poles', []): _pole(sc, x, y, spec.get('pole_radius', .06), spec.get('pole_height', .5))
            for u, v in zip(qb[:ib + 1:2, :2][:-1], qb[:ib + 1:2, :2][1:]): rm.add_line(sc, u, v, PRED_C[:3] + (1,), .01, z=.007)
            for u, v in zip(qa[:ia + 1:2, :2][:-1], qa[:ia + 1:2, :2][1:]): rm.add_line(sc, u, v, REAL_C, .012, z=.009)
            img = p.r.render().copy(); gap = np.linalg.norm(a[:2] - b[:2])
            bad = fell_t is not None and t * .02 >= fell_t
            st = f"t = {t * .02:4.1f} s   real moved {np.linalg.norm(a[:2]):.2f} m   TT-real gap {gap:.2f} m" + (f"   REAL FELL at {fell_t:.1f} s" if bad else '')
            w.append_data(np.asarray(rm.label(img, title, 'green = MuJoCo real rollout   orange (translucent) = TT prediction', st, bad)))
    return path
