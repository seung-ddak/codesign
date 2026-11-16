"""try3 PPO 공유 정책 (CALM 10,000 × 형상 20, 1억 step) 을 E0 (e0000) 에 그대로 적용, CALM 모션 하나를 접촉 설정 A (학습 설정) · D (타원+impratio10+반복50) 로 실행.
영상: A1 레퍼런스 (왼쪽, 파랑) 와 TT (오른쪽) 를 같은 장면에. TT 는 A = 원래 색, D = 반투명 주황으로 겹침.
사용: run.py [모션 이름, 기본 calm_00026]. 출력: results/<모션>.json, videos/<모션>_A1_TT_A_vs_D.mp4, report.html"""
import json, sys
from pathlib import Path
import numpy as np
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'check/slip_check/code')); sys.path.insert(0, str(ROOT / 'check/slide/code')); sys.path.insert(0, str(ROOT / 'chain/code'))
import slip_eval as se
T3 = ROOT / 'legacy/runs/ppo_tracking/try3'
CK = T3 / 'stages/000/files/checkpoints/100270080'
REF = T3 / 'references/e0000/00000011'
if not (REF / f'{(sys.argv[1] if len(sys.argv) > 1 else "calm_00026")}.npz').exists(): REF = HERE / 'references/E0/00000011'  # try3 가 E0 에서 학습하지 않은 모션은 새로 리타겟한 것
E = [0.5] * 13
M = sys.argv[1] if len(sys.argv) > 1 else 'calm_00026'
EN = {'A': 'A: pyramidal, impratio 1, iter 5 (training)', 'D': 'D: elliptic, impratio 10, iter 50'}


def render(qa, qd, a1, dt, path, ra, rd, la=None, ld=None, head=None, short=('A', 'D'), side=False):
    """side=True: 두 번째 TT 를 겹치지 않고 오른쪽에 나란히 (원래 색). False: 같은 자리에 반투명 주황으로 겹침."""
    la = la or 'TT ' + EN['A']; ld = ld or 'TT ' + EN['D']; head = head or f'try3 PPO shared policy · E0 · {M}   (same scene: A1 left, TT right)'
    import mujoco, render_slide as rs
    from PIL import Image, ImageDraw
    rm = rs.rm; W, H = rs.W, rs.H; GAP = rs.GAP
    m = rs.joint_model(np.asarray(E, float)); d = mujoco.MjData(m); d2 = mujoco.MjData(m)
    tt_bodies = {b for b in range(m.nbody) if not m.body(b).name.startswith('a1/')}
    ren = mujoco.Renderer(m, height=H, width=W, max_geom=10000)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.azimuth, cam.elevation, cam.distance = 150., -20., (3.2 if side else 2.4)
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
    qa, qd, a1 = qa.copy(), qd.copy(), a1.astype(float).copy()
    o = qa[0, :2].copy(); qa[:, :2] -= o; qd[:, :2] -= o; qa[:, 1] += GAP; qd[:, 1] += GAP + (2 * GAP if side else 0); a1[:, :2] -= a1[0, :2]; a1[:, 1] -= GAP
    n = max(len(qa), len(qd), len(a1)); stride = max(1, round(1 / dt / rs.FPS))

    def frames():
        for t in range(0, n, stride):
            ka, kt, kd = min(t, len(a1) - 1), min(t, len(qa) - 1), min(t, len(qd) - 1)
            d.qpos[:19] = qa[kt]; d.qpos[19:] = a1[ka]; mujoco.mj_forward(m, d)
            cam.lookat[:] = [(qd[kd, 0] + a1[ka, 0]) / 2, (qd[kd, 1] + a1[ka, 1]) / 2, 0.2] if side else [(qa[kt, 0] + a1[ka, 0]) / 2, (qa[kt, 1] + a1[ka, 1]) / 2, 0.2]
            ren.update_scene(d, camera=cam, scene_option=opt)
            d2.qpos[:19] = qd[kd]; d2.qpos[19:] = a1[ka]; mujoco.mj_forward(m, d2); n0 = ren.scene.ngeom
            if side:  # 나란히 모드: 가운데 TT (첫 번째) 는 주황, 오른쪽 TT 는 원래 색
                for g in range(n0):
                    gg = ren.scene.geoms[g]
                    if gg.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[gg.objid] in tt_bodies: gg.rgba[:] = (1.0, 0.55, 0.1, 1.0)
            mujoco.mjv_addGeoms(m, d2, opt, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, ren.scene)
            for g in range(n0, ren.scene.ngeom):  # 겹친 TT (D) 는 반투명 주황, 함께 추가된 A1 은 숨김
                gg = ren.scene.geoms[g]
                tt = gg.objtype == mujoco.mjtObj.mjOBJ_GEOM and m.geom_bodyid[gg.objid] in tt_bodies
                gg.rgba[:] = (gg.rgba if side else (1.0, 0.55, 0.1, 0.45)) if tt else (0, 0, 0, 0)
            rm.decorate(ren.scene, [(a1[:ka + 1:2, :2], rs.BLUE, .014)], qa[:kt + 1, :2], np.zeros((0, 2)))
            im = Image.fromarray(ren.render().copy()); dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, W, 50], fill=(15, 22, 30))
            dr.text((10, 5), head, font=rm.FONT_B, fill=(255, 255, 255))
            dr.text((10, 28), f"contact slip  {short[0]} {ra['slip_contact']:.3f}  vs  {short[1]} {rd['slip_contact']:.3f} m/s   root {short[0]} {ra['root_rmse_cm']:.1f} / {short[1]} {rd['root_rmse_cm']:.1f} cm", font=rm.FONT, fill=(190, 200, 210))
            dr.rectangle([4, 54, 600, 126], fill=(15, 22, 30))
            for y, col, txt in [(62, (51, 115, 217), 'A1 reference (left)'), (84, (255, 140, 26) if side else (200, 200, 200), la), (106, (200, 200, 200) if side else (255, 140, 26), ld)]:
                dr.rectangle([12, y, 26, y + 14], fill=col); dr.text((32, y - 2), txt, font=rm.FONT, fill=(255, 255, 255))
            fa = not ra['survived'] and t >= len(qa) - 1; fd = not rd['survived'] and t >= len(qd) - 1
            st = f"t = {t * dt:4.1f} s" + (f'   TT({short[0]}) FELL' if fa else '') + (f'   TT({short[1]}) FELL' if fd else '')
            dr.rectangle([0, H - 30, W, H], fill=(180, 55, 47) if (fa or fd) else (31, 111, 139)); dr.text((10, H - 25), st, font=rm.FONT, fill=(255, 255, 255))
            yield np.asarray(im)
    try:
        rm.encode_frames(path, frames(), width=W, height=H, fps=rs.FPS)
    finally:
        ren.close()


if __name__ == '__main__':
    cfg = {n[0]: c for n, c in se.CONFIGS.items()}
    ra, qa = se.run_ck(CK, E, REF, M, cfg['A'], multi=True, keep=True); ra.update(shape='E0', config=EN['A'])
    rd, qd = se.run_ck(CK, E, REF, M, cfg['D'], multi=True, keep=True); rd.update(shape='E0', config=EN['D'])
    json.dump([ra, rd], open(HERE / 'results' / f'{M}.json', 'w'), indent=1); print(ra); print(rd)
    z = np.load(REF / f'{M}.npz'); dt = float(z['dt'])
    out = HERE / 'videos' / f'{M}_A1_TT_A_vs_D.mp4'; render(qa, qd, z['a1_qpos'], dt, out, ra, rd); print(out)
