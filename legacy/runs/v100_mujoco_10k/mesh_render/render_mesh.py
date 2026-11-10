"""MuJoCo 실제 3D 렌더링 (GPU EGL) + 장면 안에 task 트랙을 그린 영상.

물리는 저장된 E·A 를 같은 MuJoCo 재생기로 다시 실행한다 (저장된 실제 상태와 일치해야 녹화).
왼쪽: 로봇을 따라가는 카메라. 오른쪽: 트랙 전체가 보이는 고정 카메라.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys, json, math, inspect, argparse
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/cgr-ugrad-2026/quad_ws/codesign/code/tt_dynamics")
import generate_task_mujoco as gtm  # noqa: E402
from render_task_videos import encode_frames  # noqa: E402

strict_replay = gtm.replay
_ns = dict(gtm.__dict__)
exec(inspect.getsource(gtm.replay).replace("np.min(raw_clearance) < -.002 or fallen() or support.sum() < 2", "fallen()"), _ns)
relaxed_replay = _ns["replay"]

W, H, FPS = 640, 480, 25
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
except OSError:
    FONT = FONT_B = ImageFont.load_default()

GREEN = (0.15, 0.75, 0.35, 1)
WHITE = (0.95, 0.95, 0.95, 0.9)
YELLOW = (1.0, 0.8, 0.1, 1)
ORANGE = (1.0, 0.45, 0.1, 1)
CYAN = (0.2, 0.7, 1.0, 0.8)


def track_lines(task, spec):
    """[(polyline[N,2], rgba, width)] for centerline, lane edges, gates."""
    out = []
    if task == "circle":
        r, h = spec["radius"], spec["half_width"]
        th = np.linspace(0, 2 * math.pi, 121)
        for rr, c, w in ((r, GREEN, .02), (r - h, WHITE, .012), (r + h, WHITE, .012)):
            out.append((np.stack([rr * np.sin(th), r - rr * np.cos(th)], 1), c, w))
    elif task == "zigzag":
        route = np.asarray(spec["route"]); h = spec["half_width"]
        out.append((route, GREEN, .02))
        for s in (-h, h):
            out.append((route + [0, s], WHITE, .012))
        for x, y in spec["gates"]:
            out.append((np.array([[x, y - h], [x, y + h]]), YELLOW, .025))
    elif task == "straight":
        h = spec["half_width"]
        out.append((np.array([[-.5, 0], [6, 0]]), GREEN, .02))
        for s in (-h, h):
            out.append((np.array([[-.5, s], [6, s]]), WHITE, .012))
        for x in range(1, 7):
            out.append((np.array([[x, -.3], [x, .3]]), (0.6, 0.6, 0.6, .7), .006))
    return out


def add_line(scene, a, b, rgba, width, z=0.004):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, np.array([a[0], a[1], z]), np.array([b[0], b[1], z]))
    scene.ngeom += 1


def decorate(scene, lines, trail, pred_xy):
    for pts, c, w in lines:
        for a, b in zip(pts[:-1], pts[1:]):
            add_line(scene, a, b, c, w)
    for a, b in zip(pred_xy[:-1:5], pred_xy[5::5]):
        add_line(scene, a, b, CYAN, .008, z=.006)
    for a, b in zip(trail[:-1:2], trail[2::2]):
        add_line(scene, a, b, ORANGE, .014, z=.008)


def label(img, title, sub, status, bad):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 50], fill=(15, 22, 30))
    d.text((10, 5), title, font=FONT_B, fill=(255, 255, 255))
    d.text((10, 28), sub, font=FONT, fill=(190, 200, 210))
    d.rectangle([0, H - 30, W, H], fill=(180, 55, 47) if bad else (31, 111, 139))
    d.text((10, H - 25), status, font=FONT, fill=(255, 255, 255))
    return np.asarray(im)


def render(out, *, e, token, task, title, sub_left, replay_fn, saved=None, second=None):
    """second: optional (token2, sub_right) -> left/right = two runs chase-cam (no overview)."""
    reward = gtm.TASKS[task]() if task else None
    spec = json.loads(json.dumps(reward.describe())) if reward else {}
    runs = []
    extras = [second] if isinstance(second, tuple) else (second or [])
    for tok in [token] + [x[0] for x in extras]:
        trace = []
        actual, mask, meta = replay_fn(e, tok, task=reward, state_trace=trace)
        runs.append((trace, actual, meta, tok))
    if saved is not None:
        n = int(saved[1].sum())
        if not np.allclose(runs[0][1][:n], saved[0][:n], atol=2e-5, rtol=2e-5):
            raise RuntimeError("replay differs from saved MuJoCo result")
    model = gtm.make_physics(e, reward)
    model.vis.global_.offwidth, model.vis.global_.offheight = W, H
    model.vis.headlight.ambient[:] = [.45, .45, .45]
    model.vis.headlight.diffuse[:] = [.6, .6, .6]
    model.vis.quality.shadowsize = 2048
    data = mujoco.MjData(model)
    rend = mujoco.Renderer(model, height=H, width=W, max_geom=6000)
    chase = mujoco.MjvCamera(); chase.type = mujoco.mjtCamera.mjCAMERA_FREE
    chase.azimuth, chase.elevation, chase.distance = 135., -22., 1.6
    over = mujoco.MjvCamera(); over.type = mujoco.mjtCamera.mjCAMERA_FREE
    if task == "circle":
        over.lookat[:] = [0, 1, 0]; over.distance, over.azimuth, over.elevation = 4.2, 90., -60.
    elif task == "zigzag":
        over.lookat[:] = [3, 0, 0]; over.distance, over.azimuth, over.elevation = 6.5, 90., -50.
    else:
        over.lookat[:] = [2.5, 0, 0]; over.distance, over.azimuth, over.elevation = 6., 90., -45.
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    lines = track_lines(task, spec) if task else []

    def view(run, t, cam, follow):
        trace, actual, meta, tok = run
        n = len(trace)
        k = min(t, n - 1)
        data.qpos[:] = trace[k]; mujoco.mj_forward(model, data)
        if follow:
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.2]
        rend.update_scene(data, camera=cam, scene_option=opt)
        decorate(rend.scene, lines, actual[:k + 1, :2], tok[:, :2] if task else np.zeros((0, 2)))
        img = rend.render().copy()
        fell = meta["fell"] and t >= n - 1
        if meta.get("termination_reason") == "invalid_initial_state":
            st, bad = "invalid start state", True
        elif fell:
            st, bad = f"FELL at {meta['valid_seconds']:.2f} s", True
        else:
            st, bad = f"t = {t * 0.02:4.1f} s   forward {actual[k, 0] - actual[0, 0]:+.2f} m", False
        return img, st, bad

    def frames():
        for t in range(0, 501, 50 // FPS):
            a, st, bad = view(runs[0], t, chase, True)
            a = label(a, title, sub_left, st, bad)
            panels = [a]
            for i, (_, sub) in enumerate(extras):
                b, st2, bad2 = view(runs[i + 1], t, chase, True)
                panels.append(label(b, title, sub, st2, bad2))
            if not extras:
                b, _, _ = view(runs[0], t, over, False)
                panels.append(label(b, "Track overview", "green: route  white: lane  yellow: gates  orange: robot path  blue: TT plan", st, bad))
            yield np.concatenate(panels, 1)
    out = Path(out)
    if out.exists():
        out.unlink()
    try:
        encode_frames(out, frames(), width=(1 + max(len(extras), 1)) * W, height=H, fps=FPS)
    finally:
        rend.close()
    return [dict(seconds=r[2]["valid_seconds"], reason=r[2]["termination_reason"]) for r in runs]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path)
    a = ap.parse_args()
    res = []
    for j in json.load(open(a.jobs)):
        z = np.load(j["npz"])
        if j["kind"] == "task":
            c = j["candidate"]
            r = render(j["out"], e=z["E"][c], token=z["pred"][c], task=j["task"], title=j["title"], sub_left=j["sub"],
                       replay_fn=strict_replay, saved=(z["mujoco"][c], z["valid"][c]))
        elif j["kind"] == "repro3":
            z2 = np.load(j["npz2"])
            i = list(z["ids"]).index(j["id"]); assert list(z2["ids"]) == list(z["ids"])
            true = z["true"][i].copy(); L = int(z["mask"][i].sum()); true[L:, 42:] = true[L - 1, 42:]
            r = render(j["out"], e=z["E"][i], token=true, task=None, title=j["title"], sub_left="REAL action from data",
                       replay_fn=relaxed_replay, second=[(z["mode2"][i], "H64 (before) generated action"),
                                                         (z2["mode2"][i], "New TT (delta+trim) generated action")])
        else:
            i = list(z["ids"]).index(j["id"])
            true = z["true"][i].copy(); L = int(z["mask"][i].sum()); true[L:, 42:] = true[L - 1, 42:]
            r = render(j["out"], e=z["E"][i], token=true, task=None, title=j["title"], sub_left="REAL action from data",
                       replay_fn=relaxed_replay, second=(z["mode2"][i], "TT-generated action"))
        res.append(dict(out=j["out"], result=r)); print(json.dumps(res[-1]), flush=True)
