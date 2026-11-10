"""e0000 PPO 트래킹 비교 영상 (h64_delta_track/videos 와 같은 메쉬 표현, mesh_render/render_three.py 기반).

칸 1: A1 원본 CALM 모션 (30 Hz 원본을 같은 시작 프레임부터 재생)
칸 2: TT e0000 · 형상 전용 PPO 트래킹의 실제 MuJoCo 실행 (50 Hz, rollouts/e0000)
칸 3 (--gen 이 있을 때): TT 가 같은 시작 상태에서 생성한 행동을 MuJoCo 로 재실행
실행 전 `source env_osmesa.sh`.
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
os.environ.setdefault("MUJOCO_GL", "osmesa")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "runs/v100_mujoco_10k/mesh_render"))
import render_three as r3  # noqa: E402
rm, Panel, W, H, FPS = r3.rm, r3.Panel, r3.W, r3.H, r3.FPS
DATA = ROOT / "runs/calm_dynamics_1000x100/rollouts/e0000/primitive_seed0.npz"


def render(ep, gen=None):
    z = np.load(DATA, allow_pickle=True)
    E = z["E"].astype(np.float64); n = int(z["valid"][ep].sum()); fell = bool(z["fell"][ep])
    motion = str(z["motions"][ep]); start = int(z["start_frame"][ep])
    title = f"e0000 · {motion}"
    ref = np.load(r3.MOTIONS / f"{int(motion.split('_')[1]):05d}.npz")
    am = r3.a1_model(); qa = np.array([am.jnt_qposadr[am.joint(j).id] for j in r3.A1_JOINTS])
    a1_q = []
    for k in range(len(ref["q"])):
        q = am.qpos0.copy(); q[:3] = ref["root"][k, :3]; q[3:7] = ref["root"][k, [6, 3, 4, 5]]; q[qa] = ref["q"][k]; a1_q.append(q)
    a1_q = np.asarray(a1_q); a1_q[:, :2] -= a1_q[min(start, len(a1_q) - 1), :2]
    tr_q = z["qpos"][ep, :n + 1].astype(np.float64); tr_q = tr_q - np.r_[tr_q[0, :2], np.zeros(tr_q.shape[1] - 2)]
    panels = [Panel(am), Panel(rm.gtm.make_physics(E))]
    if gen is not None:
        gen_q, gen_sec, gen_fell = gen
        gen_q = gen_q - np.r_[gen_q[0, :2], np.zeros(gen_q.shape[1] - 2)]
        panels.append(Panel(rm.gtm.make_physics(E)))
    seconds = len(z["valid"][ep]) * float(z["dt"])
    n_a1 = len(a1_q) - 1 - start
    jr, rr = np.degrees(z["tracking_joint_rmse_rad"][ep]), z["tracking_root_rmse_m"][ep]

    def status(sec, end_sec, fell_, xy):
        if fell_ and sec >= end_sec:
            return f"FELL at {end_sec:.1f} s", True
        return f"t = {sec:4.1f} s   moved {np.linalg.norm(xy):.2f} m", False

    def frames():
        for t in range(0, int(round(seconds / .02)) + 1, 50 // FPS):
            sec = t * .02
            ka = min(start + int(round(sec * 30)), len(a1_q) - 1)
            a = panels[0].frame(a1_q[ka], a1_q[start:ka + 1, :2], a1_q[ka, :2])
            st = (f"t = {sec:4.1f} s   moved {np.linalg.norm(a1_q[ka, :2]):.2f} m", False) if sec <= n_a1 / 30 + 1e-6 else (f"A1 motion ended at {n_a1 / 30:.1f} s", False)
            row = [rm.label(a, title, "A1: original CALM motion", *st)]
            kp = min(t, len(tr_q) - 1)
            b = panels[1].frame(tr_q[kp], tr_q[:kp + 1, :2], tr_q[kp, :2])
            row.append(rm.label(b, title, f"TT shape: PPO motion tracking  (joint {jr:.1f} deg, root {rr * 100:.0f} cm)",
                                *status(sec, n * .02, fell, tr_q[kp, :2])))
            if gen is not None:
                kg = min(t, len(gen_q) - 1)
                c = panels[2].frame(gen_q[kg], gen_q[:kg + 1, :2], gen_q[kg, :2])
                row.append(rm.label(c, title, "TT model: generated action (H64)", *status(sec, gen_sec, gen_fell, gen_q[kg, :2])))
            yield np.concatenate(row, 1)

    out = HERE / "videos" / f"ep{ep:03d}_{motion}{'_3panel' if gen is not None else ''}.mp4"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        out.unlink()
    try:
        rm.encode_frames(out, frames(), width=len(panels) * W, height=H, fps=FPS)
    finally:
        for p in panels:
            p.r.close()
    return dict(episode=ep, motion=motion, video=str(out.relative_to(HERE)), fell=fell, seconds=n * .02,
                joint_rmse_deg=float(jr), root_rmse_m=float(rr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, nargs="+", required=True)
    rows = [render(ep) for ep in ap.parse_args().episodes]
    for r in rows: print(json.dumps(r), flush=True)
    (HERE / "videos.json").write_text(json.dumps(rows, indent=1))
