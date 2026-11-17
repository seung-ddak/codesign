"""원본 · 30 Hz · 50 Hz · 100 Hz 정책을 한 장면에 나란히 (2026-11-17 사용자 요청). hz50/code/video_side.py 와 같은 형상·모션·평가 (evalpol, 외란·잡음 없음, 처음 10 초).
원본 = hz100 refs 레퍼런스 (100 Hz, 초록), 30 Hz = s6c50 20M (원래 색), 50 Hz = hz50 20M (파랑), 100 Hz = hz100 20M (주황). 모두 50 Hz 로 시간 보간. 각자 시작 위치 기준, 옆으로 LANE m 간격.
출력: video/<형상>_<모션>_ref_30_50_100.mp4"""
import os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'egl')
HERE = Path(__file__).resolve().parents[1]; H50 = HERE.parent / 'hz50'
sys.path.insert(0, str(H50 / 'code'))
from video import ROOT, SHAPES, PICK, TARGETS, STEPS, evalpol, s6c, ckpts
import imageio, mujoco, render_three as r3
from pd_replay import build_model
LANE = 0.7
COLORS = [(0.15, 0.75, 0.35, 1.0), None, (0.2, 0.5, 1.0, 1.0), (1.0, 0.55, 0.1, 1.0)]


def to50(q, hz):
    t = np.arange(len(q)) / hz; t2 = np.arange(0, t[-1] + 1e-9, 1 / 50)
    out = np.stack([np.interp(t2, t, q[:, j]) for j in range(q.shape[1])], 1); out[:, 3:7] /= np.linalg.norm(out[:, 3:7], axis=1, keepdims=True); return out


def ref_qpos(d, c, hz=100, sec=10):
    z = np.load(d / f'{c}.npz'); return np.c_[z['fix_base_pos'], z['fix_base_quat'], z['fix_q']][:int(sec * hz) + 1]


def render_side(qs, E, path, title, sub, status=''):
    rm = r3.rm; m = build_model('00000011', uniforms=list(E), flat_ground=True, torque_limit=33.5); p = r3.Panel(m); p.cam.distance = 3.2
    ds = [mujoco.MjData(m) for _ in qs]; qs = [np.asarray(q).copy() for q in qs]; k0 = (len(qs) - 1) / 2
    for k, q in enumerate(qs): q[:, :2] -= q[0, :2]; q[:, 1] += (k0 - k) * LANE
    n = max(len(q) for q in qs); step = max(1, 50 // r3.FPS)
    with imageio.get_writer(str(path), fps=r3.FPS, codec='libx264', quality=7, macro_block_size=8) as w:
        for t in range(0, n, step):
            cur = [q[min(t, len(q) - 1)] for q in qs]; c = np.mean([a[:2] for a in cur], 0)
            p.d.qpos[:] = cur[1]; mujoco.mj_forward(m, p.d); p.cam.lookat[:] = [c[0], c[1], 0.2]
            p.r.update_scene(p.d, camera=p.cam, scene_option=p.opt)
            for k in range(len(qs)):
                if COLORS[k] is None: continue
                ds[k].qpos[:] = cur[k]; mujoco.mj_forward(m, ds[k]); n0 = p.r.scene.ngeom
                mujoco.mjv_addGeoms(m, ds[k], p.opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, p.r.scene)
                for g in range(n0, p.r.scene.ngeom): p.r.scene.geoms[g].rgba[:] = COLORS[k]
            for q in qs: rm.decorate(p.r.scene, [], q[:min(t, len(q) - 1) + 1, :2], np.zeros((0, 2)))
            w.append_data(np.asarray(rm.label(p.r.render().copy(), title, sub, f't = {t * .02:4.1f} s   {status}', False)))
    return path


def main():
    out = HERE / 'video'; out.mkdir(exist_ok=True)
    for g, c, why in TARGETS:
        ck30 = min(ckpts(ROOT / 'chain/runs/s6c50_calm' / g / c), key=lambda k: abs(int(Path(k).name) - STEPS))
        ck50, ck100 = ckpts(H50 / 'runs' / g / c)[-1], ckpts(HERE / 'runs' / g / c)[-1]
        for s in PICK:
            E = SHAPES[s]['E']; d = '00000011'
            r30 = evalpol.evaluate(ck30, E, s6c.REFS / s / d, c, multi=True, keep_qpos=True, max_seconds=10)
            r50 = evalpol.evaluate(ck50, E, H50 / 'refs' / s / d, c, multi=True, keep_qpos=True, max_seconds=10)
            r100 = evalpol.evaluate(ck100, E, HERE / 'refs' / s / d, c, multi=True, keep_qpos=True, max_seconds=10)
            tag = lambda r: f"{r['joint_rmse_deg']:.1f}°/{r['root_rmse_cm']:.1f}cm{'' if r['survived'] else ' FELL'}"
            why_en = why.replace('30Hz ', '30 Hz ').replace('실패', 'fail').replace('통과', 'pass')
            render_side([to50(ref_qpos(HERE / 'refs' / s / d, c), 100), to50(np.asarray(r30['qpos']), 30), np.asarray(r50['qpos']), to50(np.asarray(r100['qpos']), 100)], E,
                        out / f'{s}_{c}_ref_30_50_100.mp4', title=f'{s} · {c} ({why_en})',
                        sub='green = original   white = 30 Hz   blue = 50 Hz   orange = 100 Hz',
                        status=f'30Hz {tag(r30)} | 50Hz {tag(r50)} | 100Hz {tag(r100)}')
            print(s, c, '30Hz', tag(r30), '| 50Hz', tag(r50), '| 100Hz', tag(r100), flush=True)


if __name__ == '__main__':
    main()
