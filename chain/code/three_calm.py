"""3 방향 (CCW · Forward · CW) 이 섞인 새 CALM 모션 1 개 (s6 학습에 쓰인 모션) 로 영상 3 개:
1) A1 단독  2) A1 + 학습 형상 TT (같은 장면)  3) A1 + 학습 형상 TT + 미사용 형상 TT (같은 장면).
TT 정책 = CCW 묶음 s6 공용 정책 (100M). 학습 형상 = 이 모션을 학습한 G_l+00_b+00, 미사용 = s8 형상 G_l+20_b+10.
출력: runs/s7_eval/videos/three_calm/"""
import json, os, sys
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
from common import ROOT, RUNS, log
import evalpol
from s7_eval import ckpts
sys.path.insert(0, str(ROOT / 'check/slide/code')); sys.path.insert(0, str(ROOT / 'legacy/runs/v100_mujoco_10k/mesh_render'))
M, G = 'calm_round_00_00019', 'trotCCWSlow__r00'
S5 = RUNS / 's5_retarget'; OUT = RUNS / 's7_eval/videos/three_calm'
TR = ('G_l+00_b+00', json.load(open(S5 / 'assignment.json'))['shapes']['G_l+00_b+00']['E'])
UN = ('G_l+20_b+10', json.load(open(RUNS / 's8_collect/shapes.json'))['G_l+20_b+10']['E'])
GREEN = np.array([.15, .7, .3, 1.])


def a1_only(z, path):
    import render_three as r3
    rm = r3.rm; dt = float(z['dt']); a = z['a1_qpos'].astype(float).copy(); a[:, :2] -= a[0, :2]; p = r3.Panel(r3.a1_model())

    def frames():
        for t in np.arange(0, len(a) * dt, 1 / r3.FPS):
            k = min(int(round(t / dt)), len(a) - 1)
            yield rm.label(p.frame(a[k], a[:k + 1, :2], a[k, :2]), f'{M} · CCW + Forward + CW', 'A1: original CALM motion', f't = {t:4.1f} s   moved {np.linalg.norm(a[k, :2]):.2f} m', False)
    try: rm.encode_frames(path, frames(), width=r3.W, height=r3.H, fps=r3.FPS)
    finally: p.r.close()


def scene(a1, dt, tts, path, head, sub):
    """tts = [(qpos, E, 이름, 넘어짐)]. 첫 TT 는 A1 과 한 모델 (joint_model), 둘째 TT 는 별도 모델 geom 을 같은 장면에 추가."""
    import mujoco, render_slide as rs
    from pd_replay import build_model
    from PIL import Image, ImageDraw
    rm = rs.rm; W, H, GAP = rs.W, rs.H, rs.GAP
    m = rs.joint_model(np.asarray(tts[0][1], float)); d = mujoco.MjData(m)
    extra = [(build_model('00000011', uniforms=np.asarray(E, float), flat_ground=True), q) for q, E, _, _ in tts[1:]]
    extra = [(mm, mujoco.MjData(mm), q) for mm, q in extra]
    ren = mujoco.Renderer(m, height=H, width=W, max_geom=20000)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.azimuth, cam.elevation, cam.distance = 150., -22., 2.4 + 1.0 * len(extra)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
    a1 = a1.astype(float).copy(); a1[:, :2] -= a1[0, :2]; a1[:, 1] -= GAP
    qs = []
    for i, (q, _, _, _) in enumerate(tts):
        q = q.astype(float).copy(); q[:, :2] -= q[0, :2]; q[:, 1] += GAP + 2 * GAP * i; qs.append(q)
    n = max([len(a1)] + [len(q) for q in qs]); stride = max(1, round(1 / dt / rs.FPS)); cols = [(255, 140, 26), (40, 180, 75)]

    def frames():
        for t in range(0, n, stride):
            ka = min(t, len(a1) - 1); ks = [min(t, len(q) - 1) for q in qs]
            d.qpos[:19] = qs[0][ks[0]]; d.qpos[19:] = a1[ka]; mujoco.mj_forward(m, d)
            cx = np.mean([a1[ka, :2]] + [q[k, :2] for q, k in zip(qs, ks)], 0); cam.lookat[:] = [cx[0], cx[1], .2]
            ren.update_scene(d, camera=cam, scene_option=opt)
            for (mm, dd, _), q, k in zip(extra, qs[1:], ks[1:]):
                dd.qpos[:] = q[k]; mujoco.mj_forward(mm, dd)
                mujoco.mjv_addGeoms(mm, dd, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, ren.scene)
            lines = [(a1[:ka + 1:2, :2], rs.BLUE, .014)] + [(q[:k + 1:2, :2], GREEN, .014) for q, k in zip(qs[1:], ks[1:])]
            rm.decorate(ren.scene, lines, qs[0][:ks[0] + 1, :2], np.zeros((0, 2)))
            im = Image.fromarray(ren.render().copy()); dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, W, 50], fill=(15, 22, 30)); dr.text((10, 5), head, font=rm.FONT_B, fill=(255, 255, 255)); dr.text((10, 28), sub, font=rm.FONT, fill=(190, 200, 210))
            leg = [((51, 115, 217), 'A1 reference (left)')] + [(cols[i], nm) for i, (_, _, nm, _) in enumerate(tts)]
            dr.rectangle([4, 54, 560, 60 + 22 * len(leg)], fill=(15, 22, 30))
            for i, (c, txt) in enumerate(leg): y = 62 + 22 * i; dr.rectangle([12, y, 26, y + 14], fill=c); dr.text((32, y - 2), txt, font=rm.FONT, fill=(255, 255, 255))
            fell = [nm for (q, _, nm, f), k in zip(tts, ks) if f and k >= len(q) - 1]
            st = f"t = {t * dt:4.1f} s" + (''.join(f'   {x.split(":")[0]} FELL' for x in fell))
            dr.rectangle([0, H - 30, W, H], fill=(180, 55, 47) if fell else (31, 111, 139)); dr.text((10, H - 25), st, font=rm.FONT, fill=(255, 255, 255))
            yield np.asarray(im)
    try: rm.encode_frames(path, frames(), width=W, height=H, fps=rs.FPS)
    finally: ren.close()


def main():
    import pd_replay as pd
    OUT.mkdir(parents=True, exist_ok=True); ck = ckpts(RUNS / 's6_train' / G)[-1]; res = {}; runs = []
    dtr = S5 / 'references' / TR[0] / '00000011'; dun = OUT / 'refs' / UN[0]
    if not (dun / '00000011' / f'{M}.npz').exists(): pd.retarget_tt(str(S5 / 'a1_reference'), str(dun), only=[M], choices=['00000011'], uniforms=UN[1], torque_limit=33.5)
    for (s, E), d, kind in [(TR, dtr, 'train'), (UN, dun / '00000011', 'unseen')]:
        z = np.load(d / f'{M}.npz'); dt = float(z['dt'])
        r = evalpol.evaluate(ck, E, d, M, multi=True, keep_qpos=True, max_seconds=len(z['fix_q']) * dt)
        runs.append((np.asarray(r.pop('qpos')), E, f"TT {kind} {s}: joint {r['joint_rmse_deg']:.1f} deg, root {r['root_rmse_cm']:.1f} cm", not r['survived']))
        res[kind] = dict(shape=s, passed=evalpol.passes(r), **{k: v for k, v in r.items() if np.isscalar(v) or v is None}); log(f"three {kind} {s} 생존 {r['survived']} 관절 {r['joint_rmse_deg']:.1f}°")
    z = np.load(dtr / f'{M}.npz'); dt = float(z['dt'])
    a1_only(z, OUT / f'1_{M}_A1_only.mp4')
    scene(z['a1_qpos'], dt, runs[:1], OUT / f'2_{M}_A1_vs_TT_train.mp4', f'{M} · CCW policy (100M) · same scene', 'A1 vs TT trained shape')
    scene(z['a1_qpos'], dt, runs, OUT / f'3_{M}_A1_vs_TT_train_vs_unseen.mp4', f'{M} · CCW policy (100M) · same scene', 'A1 vs TT trained shape vs TT unseen shape')
    json.dump(dict(motion=M, policy=G, checkpoint=str(ck), **res), open(OUT / 'three_calm.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
