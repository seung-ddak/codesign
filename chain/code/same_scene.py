"""새 CALM 모션 (체인 s4) 을 A1 (원본) 과 TT 형상 (s6 공용 정책) 에 같은 장면에 나란히 두고 녹화. 묶음마다 1 개:
s7 에서 통과한 CALM 중 가장 긴 모션, 그 평가 형상. 파랑 궤적 = A1, 주황 = TT.
출력: runs/s7_eval/videos/same_scene/<묶음>_<모션>_<형상>.mp4, same_scene.json"""
import json, os, sys
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
from common import ROOT, RUNS, log
import evalpol
sys.path.insert(0, str(ROOT / 'check/slide/code'))
S7 = RUNS / 's7_eval'; S5 = RUNS / 's5_retarget'; OUT = S7 / 'videos/same_scene'
SH = {**json.load(open(S5 / 'assignment.json'))['shapes']}


def refdir(s, c):
    for d in (S5 / 'references' / s / '00000011', S7 / 'unpaired_refs' / s / '00000011'):
        if (d / f'{c}.npz').exists(): return d


def render(q, a1, dt, E, path, head, sub, fell):
    import mujoco, render_slide as rs
    from PIL import Image, ImageDraw
    rm = rs.rm; W, H, GAP = rs.W, rs.H, rs.GAP
    m = rs.joint_model(np.asarray(E, float)); d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=H, width=W, max_geom=10000)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.azimuth, cam.elevation, cam.distance = 150., -20., 2.4
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
    q, a1 = q.astype(float).copy(), a1.astype(float).copy(); q[:, :2] -= q[0, :2]; a1[:, :2] -= a1[0, :2]; q[:, 1] += GAP; a1[:, 1] -= GAP
    n = max(len(q), len(a1)); stride = max(1, round(1 / dt / rs.FPS))

    def frames():
        for t in range(0, n, stride):
            ka, kt = min(t, len(a1) - 1), min(t, len(q) - 1)
            d.qpos[:19] = q[kt]; d.qpos[19:] = a1[ka]; mujoco.mj_forward(m, d)
            cam.lookat[:] = [(q[kt, 0] + a1[ka, 0]) / 2, (q[kt, 1] + a1[ka, 1]) / 2, .2]
            ren.update_scene(d, camera=cam, scene_option=opt)
            rm.decorate(ren.scene, [(a1[:ka + 1:2, :2], rs.BLUE, .014)], q[:kt + 1, :2], np.zeros((0, 2)))
            im = Image.fromarray(ren.render().copy()); dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, W, 50], fill=(15, 22, 30)); dr.text((10, 5), head, font=rm.FONT_B, fill=(255, 255, 255)); dr.text((10, 28), sub, font=rm.FONT, fill=(190, 200, 210))
            f = fell and t >= len(q) - 1
            st = f"t = {t * dt:4.1f} s   A1 moved {np.linalg.norm(a1[ka, :2] - a1[0, :2]):.2f} m   TT moved {np.linalg.norm(q[kt, :2] - q[0, :2]):.2f} m" + ('   TT FELL' if f else '')
            dr.rectangle([0, H - 30, W, H], fill=(180, 55, 47) if f else (31, 111, 139)); dr.text((10, H - 25), st, font=rm.FONT, fill=(255, 255, 255))
            yield np.asarray(im)
    try: rm.encode_frames(path, frames(), width=W, height=H, fps=rs.FPS)
    finally: ren.close()


def main():
    from s7_eval import ckpts
    r7 = json.load(open(S7 / 'results.json')); OUT.mkdir(parents=True, exist_ok=True); rows = []
    for g, a in r7.items():
        ok = [x for x in a['results'] if x['clip'] != g and evalpol.passes(x) and x['shape'] in SH and refdir(x['shape'], x['clip'])]
        if not ok: log(f'{g}: 통과한 CALM 없음'); continue
        x = max(ok, key=lambda x: x['clip_seconds']); s, c = x['shape'], x['clip']; E = SH[s]['E']
        ck = a.get('checkpoint') or ckpts(RUNS / 's6_train' / g)[-1]; rd = refdir(s, c)
        r = evalpol.evaluate(ck, E, rd, c, multi=True, keep_qpos=True, max_seconds=10.0)
        z = np.load(rd / f'{c}.npz'); dt = float(z['dt'])
        gn = g.replace('trot', '').replace('Slow__r00', '')
        dst = OUT / f'{gn}_{c}_{s}.mp4'
        render(np.asarray(r['qpos']), z['a1_qpos'], dt, E, dst, f'{c} · {gn} policy · TT {s}   (same scene: A1 left, TT right)',
               f"joint RMSE {r['joint_rmse_deg']:.1f} deg   root RMSE {r['root_rmse_cm']:.1f} cm   slip {r['slip_m_s']:.3f} m/s", not r['survived'])
        rows.append(dict(group=g, clip=c, shape=s, video=dst.name, survived=r['survived'], seconds=r['seconds'], joint_rmse_deg=r['joint_rmse_deg'], root_rmse_cm=r['root_rmse_cm'], passed=evalpol.passes(r)))
        log(f'same_scene {dst.name} 관절 {r["joint_rmse_deg"]:.1f}° 통과 {rows[-1]["passed"]}')
    json.dump(rows, open(OUT / 'same_scene.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
