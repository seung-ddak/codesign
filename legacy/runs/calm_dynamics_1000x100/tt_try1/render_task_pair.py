"""task 별 reward gradient (왼쪽) | RL (오른쪽) 실제 MuJoCo 재실행 영상. mesh_render/render_mesh.py 와 같은 표현.

각 방법의 최고 보상 후보를 고른다. 두 방법은 형상 E 가 다를 수 있어 칸마다 자기 형상 모델로 그린다.
저장된 실제 결과와 재실행이 다르면 멈춘다. 실행 전 `source env_osmesa.sh`.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
os.environ.setdefault("MUJOCO_GL", "osmesa")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "runs/v100_mujoco_10k/mesh_render"))
import render_mesh as rm  # noqa: E402
import mujoco  # noqa: E402
W, H, FPS, gtm = rm.W, rm.H, rm.FPS, rm.gtm
NAMES = {"circle": "Circle track", "zigzag": "Pillar slalom", "straight": "Straight (max speed)"}
METHODS = [("gradient", "reward gradient denoising"), ("rl_ppo", "RL denoising (PPO)")]


def best(root, task, method):
    """method 이 'gradient' 면 모든 유도 강도 (gradient_scale_*) 중, 모든 seed 중 최고 보상 후보."""
    dirs = sorted((root / task).glob("gradient_scale_*")) if method == "gradient" else [root / task / method]
    rows = []
    for d in dirs:
        for f in sorted(d.glob("seed_*.json")):
            js = json.load(open(f))
            rows += [(f, i, c, d.name) for i, c in enumerate(js["candidates"]) if c.get("reward") is not None]
    f, i, c, name = max(rows, key=lambda x: x[2]["reward"])
    z = np.load(f.with_suffix(".npz"))
    return dict(e=z["E"][i].astype(np.float64), token=z["pred"][i], saved=(z["mujoco"][i], z["valid"][i]), reward=c["reward"], cand=i, source=f"{name}/{f.stem}")


class Run:
    def __init__(self, task, b, sub):
        self.reward = gtm.TASKS[task](); self.task, self.sub, self.tok = task, sub, b["token"]
        trace = []
        self.actual, _, self.meta = rm.strict_replay(b["e"], b["token"], task=self.reward, state_trace=trace)
        n = int(b["saved"][1].sum())
        if not np.allclose(self.actual[:n], b["saved"][0][:n], atol=2e-5, rtol=2e-5):
            raise RuntimeError("replay differs from saved MuJoCo result")
        self.trace = trace
        m = gtm.make_physics(b["e"], self.reward)
        m.vis.global_.offwidth, m.vis.global_.offheight = W, H
        m.vis.headlight.ambient[:] = [.45, .45, .45]; m.vis.headlight.diffuse[:] = [.6, .6, .6]; m.vis.quality.shadowsize = 2048
        self.m, self.d = m, mujoco.MjData(m)
        self.r = mujoco.Renderer(m, height=H, width=W, max_geom=6000)
        self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.elevation, self.cam.distance = 135., -22., 1.6
        self.opt = mujoco.MjvOption(); self.opt.geomgroup[:] = 1
        self.lines = rm.track_lines(task, json.loads(json.dumps(self.reward.describe())))
        scale = b["source"].split("/")[0].replace("gradient_scale_", "scale ") if b["source"].startswith("gradient") else ""
        self.sub = f"{sub} {scale}   reward {b['reward']:.2f}".replace("  ", " ")

    def frame(self, t):
        n = len(self.trace); k = min(t, n - 1)
        self.d.qpos[:] = self.trace[k]; mujoco.mj_forward(self.m, self.d)
        self.cam.lookat[:] = [self.d.qpos[0], self.d.qpos[1], .2]
        self.r.update_scene(self.d, camera=self.cam, scene_option=self.opt)
        rm.decorate(self.r.scene, self.lines, self.actual[:k + 1, :2], self.tok[:, :2])
        img = self.r.render().copy()
        if self.meta["fell"] and t >= n - 1:
            st, bad = f"FELL at {self.meta['valid_seconds']:.2f} s", True
        else:
            st, bad = f"t = {t * gtm.DT:4.1f} s   forward {self.actual[k, 0] - self.actual[0, 0]:+.2f} m", False
        return rm.label(img, NAMES[self.task], self.sub, st, bad)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=HERE / "task_v1"); ap.add_argument("--tag", default="")
    a = ap.parse_args(); out_rows = []
    for task in ("circle", "zigzag", "straight"):
        runs = [Run(task, best(a.root, task, m), sub) for m, sub in METHODS]
        out = a.root / "video" / f"task_{task}_gradient_vs_rl{a.tag}.mp4"; out.parent.mkdir(exist_ok=True)
        if out.exists(): out.unlink()
        stride = max(1, round(1 / gtm.DT / FPS))
        def frames():
            for t in range(0, gtm.STEPS + 1, stride):
                yield np.concatenate([r.frame(t) for r in runs], 1)
        try:
            rm.encode_frames(out, frames(), width=2 * W, height=H, fps=FPS)
        finally:
            for r in runs: r.r.close()
        out_rows.append(dict(task=task, video=str(out.relative_to(HERE)),
                             runs=[dict(method=m, seconds=r.meta["valid_seconds"], fell=bool(r.meta["fell"])) for (m, _), r in zip(METHODS, runs)]))
        print(json.dumps(out_rows[-1]), flush=True)
    (a.root / "video" / "videos.json").write_text(json.dumps(out_rows, indent=1))


if __name__ == "__main__":
    main()
