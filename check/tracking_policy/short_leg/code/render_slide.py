"""평가 결과 (results/<E>/<motion>.npz) 를 영상으로: A1 레퍼런스 모션과 TT 형상 PPO 트래킹을 한 장면(같은 바닥·조명·카메라)에 나란히 둔다.
TT 장면(build_model) 에 A1 MJCF 를 붙여 하나의 MuJoCo 모델로 렌더링. A1 은 -y (화면 왼쪽), TT 는 +y (화면 오른쪽) 으로 GAP 만큼 띄운다.
출력: video/<E>/<motion>.mp4. 실행 전 `source code/env_osmesa.sh`."""
import json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import mujoco
from PIL import Image, ImageDraw
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관
sys.path.insert(0, str(ROOT / 'code/viz')); sys.path.insert(0, str(ROOT / 'code/motion_tracking')); sys.path.insert(0, str(CODE_DIR))
import render_three as r3
from pd_replay import build_model
from shapes import SHAPES, MOTIONS
rm, FPS = r3.rm, r3.FPS
W, H = 2 * r3.W, r3.H
A1_XML = r3.A1_SCENE.parent / 'a1.xml'
GAP = 0.35  # 두 로봇 몸통 중심의 y 간격의 절반 (m)
BLUE = np.array([.2, .45, .85, 1.])
EN = {k: v['en'] for k, v in SHAPES.items()}
summary = {(r['shape'], r['motion']): r for r in json.load(open(HERE / 'results/summary.json'))}


def joint_model(E):
    """TT 형상 장면 + A1. qpos = [TT 19, A1 19]."""
    def attach(spec):
        a1 = mujoco.MjSpec.from_file(str(A1_XML))
        a1.compiler.meshdir = str(A1_XML.parent / 'assets'); a1.compiler.texturedir = a1.compiler.meshdir
        for k in list(a1.keys): a1.delete(k)
        spec.worldbody.add_frame().attach_body(a1.worldbody.first_body(), 'a1/', '')
    m = build_model('00000011', uniforms=E, flat_ground=True, scene_fn=attach)
    assert m.nq == 38, m.nq
    m.vis.global_.offwidth, m.vis.global_.offheight = W, H
    m.vis.headlight.ambient[:] = [.45, .45, .45]; m.vis.headlight.diffuse[:] = [.6, .6, .6]
    return m


def label(img, title, sub, status, bad):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 50], fill=(15, 22, 30))
    d.text((10, 5), title, font=rm.FONT_B, fill=(255, 255, 255))
    d.text((10, 28), sub, font=rm.FONT, fill=(190, 200, 210))
    d.rectangle([4, 54, 250, 104], fill=(15, 22, 30))
    d.rectangle([12, 62, 26, 76], fill=(51, 115, 217)); d.text((32, 60), 'A1 reference (left)', font=rm.FONT, fill=(255, 255, 255))
    d.rectangle([12, 84, 26, 98], fill=tuple(int(255 * c) for c in rm.ORANGE[:3])); d.text((32, 82), 'TT PPO tracking (right)', font=rm.FONT, fill=(255, 255, 255))
    d.rectangle([0, H - 30, W, H], fill=(180, 55, 47) if bad else (31, 111, 139))
    d.text((10, H - 25), status, font=rm.FONT, fill=(255, 255, 255))
    return np.asarray(im)


def render(K, M):
    z = np.load(HERE / 'results' / K / f'{M}.npz'); r = summary[(K, M)]
    tt = z['qpos'].astype(np.float64); tt[:, :2] -= tt[0, :2]; tt[:, 1] += GAP
    a1 = z['a1_qpos'].astype(np.float64); a1[:, :2] -= a1[0, :2]; a1[:, 1] -= GAP
    m = joint_model(z['E'].astype(np.float64)); d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=H, width=W, max_geom=8000)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth, cam.elevation, cam.distance = 150., -20., 2.4
    opt = mujoco.MjvOption(); opt.geomgroup[:] = 1
    dt = float(z['dt']); n = len(tt) - 1; fell = not r['survived']
    title = f"{EN[K]} · {M.replace('__r00', '')}   (same scene: A1 left, TT right)"
    sub = f"TT  slip {r['slip_m_s']:.3f} m/s (ref {r['ref_slip_m_s']:.3f})   joint {r['joint_rmse_deg']:.1f} deg   root {r['root_rmse_cm']:.1f} cm"
    stride = max(1, round(1 / dt / FPS))

    def frames():
        for t in range(0, len(a1), stride):
            ka = min(t, len(a1) - 1); kt = min(t, n)
            d.qpos[:19] = tt[kt]; d.qpos[19:] = a1[ka]; mujoco.mj_forward(m, d)
            cam.lookat[:] = [(tt[kt, 0] + a1[ka, 0]) / 2, (tt[kt, 1] + a1[ka, 1]) / 2, 0.2]
            ren.update_scene(d, camera=cam, scene_option=opt)
            rm.decorate(ren.scene, [(a1[:ka + 1:2, :2], BLUE, .014)], tt[:kt + 1, :2], np.zeros((0, 2)))
            st = (f'TT FELL at {n * dt:.1f} s   t = {t * dt:4.1f} s', True) if fell and t >= n else \
                 (f't = {t * dt:4.1f} s   A1 moved {np.linalg.norm(a1[ka, :2] - a1[0, :2]):.2f} m   TT moved {np.linalg.norm(tt[kt, :2] - tt[0, :2]):.2f} m', False)
            yield label(ren.render().copy(), title, sub, *st)

    out = HERE / 'video' / K / f'{M}.mp4'; out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists(): out.unlink()
    try:
        rm.encode_frames(out, frames(), width=W, height=H, fps=FPS)
    finally:
        ren.close()
    return out


if __name__ == '__main__':
    only = sys.argv[1:]  # 예: E0/squat__r00
    for K in SHAPES:
        for M in MOTIONS:
            if (K, M) in summary and (not only or f'{K}/{M}' in only):
                print(render(K, M), flush=True)
