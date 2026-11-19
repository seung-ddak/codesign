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


def render_task_overlay(q_real, q_pred, E, task, path, title='', fell_t=None, pred_reset_s=None, fade_s=0.06, show_pred=True):
    """과제 환경 (트랙 경계·슬라럼 폴) 한 장면에 실제 MuJoCo 재생 (초록) 과 TT 예측 (주황 반투명) 을 겹쳐 그림.
    q_* = 50 Hz qpos (s0 기준 world 좌표, 원점 출발). 카메라는 두 몸통 중점을 따라감, 아래에 두 위치 차이 (m).
    pred_reset_s (2026-11-19 사용자 요청): MPC 예측이 이 주기 (s) 마다 실제 상태에서 다시 시작할 때, 경계 앞뒤 fade_s 동안 주황 예측을 투명하게 줄여
    순간 이동이 보이지 않게 하고, 주황 경로선도 경계를 건너는 구간은 그리지 않는다 (한 계획 안에서만 이어 그림).
    show_pred=False (2026-11-19 사용자 요청): TT 예측 없이 MuJoCo 실제 로봇만 그림 (예측 오차가 작아 겹쳐 그릴 필요 없음)."""
    import imageio, mujoco, render_three as r3
    from pd_replay import build_model
    # 폴은 MuJoCo 실행과 같은 geom 으로 모델에 넣는다 (2026-11-19: 이전 _pole 은 원기둥 size 를 잘못 넣어 높이 0.12 m 원판이 공중 0.41 m 에 떠 보였음)
    sys.path.insert(0, str(ROOT / 'code/tt_dynamics')); from generate_task_mujoco import add_slalom_poles
    from functools import partial
    scene_fn = partial(add_slalom_poles, task=task) if hasattr(task, 'poles') else None
    if hasattr(task, 'add_scene'): scene_fn = task.add_scene  # 2026-11-19 지형 과제 (계단 · 허들)
    rm = r3.rm; m = build_model('00000011', uniforms=list(E), flat_ground=True, torque_limit=33.5, scene_fn=scene_fn); p = r3.Panel(m); d2 = mujoco.MjData(m)
    p.cam.distance, p.cam.elevation = 2.6, -32.
    spec = task.describe(); lines = task.guide_lines()
    qa, qb = np.asarray(q_real), np.asarray(q_pred); n = max(len(qa), len(qb)); step = max(1, 50 // r3.FPS)
    with imageio.get_writer(str(path), fps=r3.FPS, codec='libx264', quality=7, macro_block_size=8) as w:
        for t in range(0, n, step):
            ia, ib = min(t, len(qa) - 1), min(t, len(qb) - 1); a, b = qa[ia], qb[ib]
            p.d.qpos[:] = a; mujoco.mj_forward(p.m, p.d)
            p.cam.lookat[:] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.15] if show_pred else [a[0], a[1], 0.15]
            p.r.update_scene(p.d, camera=p.cam, scene_option=p.opt); sc = p.r.scene
            _tint(m, sc, 0, REAL_C)
            d2.qpos[:] = b; mujoco.mj_forward(m, d2); n0 = sc.ngeom
            mujoco.mjv_addGeoms(m, d2, p.opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, sc)
            fa = 1.0 if show_pred else 0.0
            if pred_reset_s and show_pred:
                ph = (t * .02) % pred_reset_s; fa = float(np.clip(min(ph, pred_reset_s - ph) / fade_s, 0, 1))
            if fa < 1: sc.ngeom = n0 if fa == 0 else sc.ngeom
            _tint(m, sc, n0, PRED_C[:3] + (PRED_C[3] * fa,))
            for pts, solid in lines:
                for u, v in zip(pts[:-1:2], pts[2::2]): rm.add_line(sc, u, v, (1, 1, 1, .9) if solid else (.2, .3, .5, .9), .006 if solid else .012)
            tb = np.arange(0, ib + 1, 2) * .02; seg = (tb // pred_reset_s) if pred_reset_s else np.zeros_like(tb)
            for j, (u, v) in enumerate(zip(qb[:ib + 1:2, :2][:-1], qb[:ib + 1:2, :2][1:])):
                if show_pred and seg[j] == seg[j + 1]: rm.add_line(sc, u, v, PRED_C[:3] + (1,), .01, z=.007)
            for u, v in zip(qa[:ia + 1:2, :2][:-1], qa[:ia + 1:2, :2][1:]): rm.add_line(sc, u, v, REAL_C, .012, z=.009)
            img = p.r.render().copy(); gap = np.linalg.norm(a[:2] - b[:2])
            bad = fell_t is not None and t * .02 >= fell_t
            st = (f"t = {t * .02:4.1f} s   real moved {np.linalg.norm(a[:2]):.2f} m" + (f"   TT-real gap {gap:.2f} m" if show_pred else '')) + (f"   REAL FELL at {fell_t:.1f} s" if bad else '')
            w.append_data(np.asarray(rm.label(img, title, 'green = MuJoCo real rollout   orange (translucent) = TT prediction' if show_pred else 'MuJoCo real rollout (generated morphology + generated actions)', st, bad)))
    return path
