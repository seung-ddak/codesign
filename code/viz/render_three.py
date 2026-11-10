"""3-panel repro video: A1 original CALM motion | TT shape PD tracking (data) | TT-generated action.

Panel 1: A1 model, kinematic playback of the original CALM A1 rollout (30 Hz), from
         the same start frame the TT episode used.
Panel 2: TT model, kinematic playback of the stored PD-tracking MuJoCo qpos (50 Hz).
Panel 3: TT model, the TT-generated action (samples.npz mode2) re-run in MuJoCo
         with the same relaxed replay as render_mesh.py.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys, json, argparse
from pathlib import Path
import numpy as np
import mujoco

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_mesh as rm  # noqa: E402

W, H, FPS = rm.W, rm.H, rm.FPS
WS = Path("/home/cgr-ugrad-2026/quad_ws")
ROLLOUTS = WS / "codesign/legacy/runs/local_mujoco_10k_20261109/rollouts"
MOTIONS = WS / "codesign/legacy/runs/calm_motion_10,000/dataset/accepted"
A1_SCENE = WS / "transformer-transformer/assets/mjcf/unitree_a1/scene.xml"
A1_JOINTS = [f"{l}_{p}_joint" for l in ("FL", "FR", "RL", "RR") for p in ("hip", "thigh", "calf")]


def episode_source(i):
    files = []
    for d in sorted(ROLLOUTS.glob("e*")):
        files += [d / "variant_0.npz", d / "variant_1.npz"]
    f, r = files[i // 100], i % 100
    with np.load(f) as z:
        L = int(z["valid"][r].sum())
        return dict(file=f, row=r, motion=str(z["motions"][r]), start=int(z["start_frame"][r]),
                    qpos=z["qpos"][r, :L + 1].astype(np.float64), E=z["E"].astype(np.float64),
                    fell=bool(z["fell"][r]), design=f.parent.name)


class Panel:
    def __init__(self, model):
        model.vis.global_.offwidth, model.vis.global_.offheight = W, H
        model.vis.headlight.ambient[:] = [.45, .45, .45]
        model.vis.headlight.diffuse[:] = [.6, .6, .6]
        self.m, self.d = model, mujoco.MjData(model)
        self.r = mujoco.Renderer(model, height=H, width=W, max_geom=6000)
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 135., -22., 1.6
        self.opt = mujoco.MjvOption(); self.opt.geomgroup[:] = 1

    def frame(self, qpos, trail, follow_xy):
        self.d.qpos[:] = qpos; mujoco.mj_forward(self.m, self.d)
        self.cam.lookat[:] = [follow_xy[0], follow_xy[1], 0.2]
        self.r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        rm.decorate(self.r.scene, [], trail, np.zeros((0, 2)))
        return self.r.render().copy()


def a1_model():
    spec = mujoco.MjSpec.from_file(str(A1_SCENE))
    spec.compiler.meshdir = str(A1_SCENE.parent / "assets"); spec.compiler.texturedir = spec.compiler.meshdir
    return spec.compile()


def render(job):
    src = episode_source(job["id"])
    z = np.load(job["npz"])
    i = list(z["ids"]).index(job["id"])
    E = z["E"][i].astype(np.float64)
    assert np.allclose(E, src["E"], atol=1e-6), "episode/shape mismatch"
    # Panel 1: A1 original motion, 30 Hz frames from the same start frame.
    ref = np.load(MOTIONS / f"{int(src['motion'].split('_')[1]):05d}.npz")
    am = a1_model()
    qa = np.array([am.jnt_qposadr[am.joint(n).id] for n in A1_JOINTS])
    a1_q = []
    for k in range(len(ref["q"])):
        q = am.qpos0.copy()
        q[:3] = ref["root"][k, :3]; q[3:7] = ref["root"][k, [6, 3, 4, 5]]; q[qa] = ref["q"][k]
        a1_q.append(q)
    a1_q = np.asarray(a1_q)
    a1_q[:, :2] -= a1_q[src["start"], :2]
    # Panel 3: TT-generated action replayed in MuJoCo (same as earlier videos).
    trace = []
    gen_tok = z["mode2"][i]
    _, _, meta = rm.relaxed_replay(E, gen_tok, task=None, state_trace=trace)
    gen_q = np.asarray(trace)
    tt_model = rm.gtm.make_physics(E)
    p1, p2, p3 = Panel(am), Panel(tt_model), Panel(rm.gtm.make_physics(E))
    pd_q = src["qpos"]
    pd_q = pd_q - np.r_[pd_q[0, :2], np.zeros(pd_q.shape[1] - 2)]
    gen_q = gen_q - np.r_[gen_q[0, :2], np.zeros(gen_q.shape[1] - 2)]
    n_a1 = len(a1_q) - 1 - src["start"]
    gen_sec = meta["valid_seconds"]

    def status(sec, end_sec, fell, xy):
        if fell and sec >= end_sec:
            return f"FELL at {end_sec:.1f} s", True
        return f"t = {sec:4.1f} s   moved {np.linalg.norm(xy):.2f} m", False

    def frames():
        for t in range(0, 501, 50 // FPS):      # 50 Hz index, 10 s
            sec = t * .02
            ka = min(src["start"] + int(round(sec * 30)), len(a1_q) - 1)
            a = p1.frame(a1_q[ka], a1_q[src["start"]:ka + 1, :2], a1_q[ka, :2])
            st = (f"t = {sec:4.1f} s   moved {np.linalg.norm(a1_q[ka, :2]):.2f} m", False) if sec <= n_a1 / 30 + 1e-6 \
                else (f"A1 motion ended at {n_a1 / 30:.1f} s", False)
            a = rm.label(a, job["title"], "A1: original CALM motion", *st)
            kp = min(t, len(pd_q) - 1)
            b = p2.frame(pd_q[kp], pd_q[:kp + 1, :2], pd_q[kp, :2])
            b = rm.label(b, job["title"], "TT shape: motion tracking (PD data)",
                         *status(sec, (len(pd_q) - 1) * .02, src["fell"], pd_q[kp, :2]))
            kg = min(t, len(gen_q) - 1)
            c = p3.frame(gen_q[kg], gen_q[:kg + 1, :2], gen_q[kg, :2])
            c = rm.label(c, job["title"], "TT model: generated action (H64)",
                         *status(sec, gen_sec, meta["fell"], gen_q[kg, :2]))
            yield np.concatenate([a, b, c], 1)

    out = Path(job["out"])
    if out.exists():
        out.unlink()
    try:
        rm.encode_frames(out, frames(), width=3 * W, height=H, fps=FPS)
    finally:
        for p in (p1, p2, p3):
            p.r.close()
    return dict(id=job["id"], design=src["design"], motion=src["motion"], start_frame=src["start"],
                pd_seconds=(len(pd_q) - 1) * .02, pd_fell=src["fell"],
                gen_seconds=gen_sec, gen_fell=bool(meta["fell"]),
                a1_disp=float(np.linalg.norm(a1_q[min(src["start"] + 300, len(a1_q) - 1), :2])),
                pd_disp=float(np.linalg.norm(pd_q[-1, :2])), gen_disp=float(np.linalg.norm(gen_q[-1, :2])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", type=Path)
    for j in json.load(open(ap.parse_args().jobs)):
        print(json.dumps(render(j)), flush=True)
