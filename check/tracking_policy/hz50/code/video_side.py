"""원본 모션 · 30 Hz · 50 Hz 정책을 한 장면에 나란히 (2026-11-17 사용자 요청). video.py 와 같은 형상·모션·평가 (evalpol, 외란·잡음 없음, 처음 10 초).
원본 = refs/<형상>/00000011 레퍼런스 (50 Hz, 초록), 30 Hz = s6c50 20M (원래 색, 50 Hz 로 보간), 50 Hz = hz50 20M (파랑). 각자 시작 위치 기준, 옆으로 LANE m 간격.
출력: video/<형상>_<모션>_ref_30_50.mp4"""
import os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from video import ROOT, HERE, SHAPES, PICK, TARGETS, STEPS, to50, evalpol, s6c, ckpts
import imageio, mujoco, render_three as r3
from pd_replay import build_model
LANE = 0.7
LANES = [('original', (0.15, 0.75, 0.35, 1.0)), ('30 Hz', None), ('50 Hz', (0.2, 0.5, 1.0, 1.0))]


def ref_qpos(d, c, sec=10):
    z = np.load(d / f'{c}.npz'); return np.c_[z['fix_base_pos'], z['fix_base_quat'], z['fix_q']][:int(sec * 50) + 1]


def render_side(qs, E, path, title, sub, status=''):
    rm = r3.rm; m = build_model('00000011', uniforms=list(E), flat_ground=True, torque_limit=33.5); p = r3.Panel(m); p.cam.distance = 2.6
    ds = [mujoco.MjData(m) for _ in qs]; qs = [np.asarray(q).copy() for q in qs]
    for k, q in enumerate(qs): q[:, :2] -= q[0, :2]; q[:, 1] += (1 - k) * LANE
    n = max(len(q) for q in qs); step = max(1, 50 // r3.FPS)
    with imageio.get_writer(str(path), fps=r3.FPS, codec='libx264', quality=7, macro_block_size=8) as w:
        for t in range(0, n, step):
            cur = [q[min(t, len(q) - 1)] for q in qs]; c = np.mean([a[:2] for a in cur], 0)
            p.d.qpos[:] = cur[1]; mujoco.mj_forward(m, p.d); p.cam.lookat[:] = [c[0], c[1], 0.2]
            p.r.update_scene(p.d, camera=p.cam, scene_option=p.opt)
            for k in (0, 2):
                ds[k].qpos[:] = cur[k]; mujoco.mj_forward(m, ds[k]); n0 = p.r.scene.ngeom
                mujoco.mjv_addGeoms(m, ds[k], p.opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, p.r.scene)
                for g in range(n0, p.r.scene.ngeom): p.r.scene.geoms[g].rgba[:] = LANES[k][1]
            for q in qs: rm.decorate(p.r.scene, [], q[:min(t, len(q) - 1) + 1, :2], np.zeros((0, 2)))
            w.append_data(np.asarray(rm.label(p.r.render().copy(), title, sub, f't = {t * .02:4.1f} s   {status}', False)))
    return path


def main():
    out = HERE / 'video'; out.mkdir(exist_ok=True)
    for g, c, why in TARGETS:
        ck30 = min(ckpts(ROOT / 'chain/runs/s6c50_calm' / g / c), key=lambda k: abs(int(Path(k).name) - STEPS)); ck50 = ckpts(HERE / 'runs' / g / c)[-1]
        for s in PICK:
            E = SHAPES[s]['E']
            r30 = evalpol.evaluate(ck30, E, s6c.REFS / s / '00000011', c, multi=True, keep_qpos=True, max_seconds=10)
            r50 = evalpol.evaluate(ck50, E, HERE / 'refs' / s / '00000011', c, multi=True, keep_qpos=True, max_seconds=10)
            tag = lambda r: f"{r['joint_rmse_deg']:.1f} deg / {r['root_rmse_cm']:.1f} cm{'' if r['survived'] else ' FELL'}"
            why_en = why.replace('30Hz ', '30 Hz ').replace('실패', 'fail').replace('통과', 'pass')
            render_side([ref_qpos(HERE / 'refs' / s / '00000011', c), to50(np.asarray(r30['qpos']), 30), np.asarray(r50['qpos'])], E,
                        out / f'{s}_{c}_ref_30_50.mp4', title=f'{s} · {c} ({why_en})',
                        sub='green = original motion   white = 30 Hz policy   blue = 50 Hz policy',
                        status=f'30 Hz {tag(r30)}   |   50 Hz {tag(r50)}')
            print(s, c, '30Hz', tag(r30), '| 50Hz', tag(r50), flush=True)


if __name__ == '__main__':
    main()
