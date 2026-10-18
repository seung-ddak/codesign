"""7 단계: 학습한 TT 로 복합 task 의 형상 E·state S·action A 를 reward guidance 로 생성하고, E·A 를 MJX 에서 open-loop 실행한다.

생성: window (H step) 를 이어 붙여 10 s (500 step). 첫 window 는 E 도 노이즈에서 생성 (joint 조건), 이후 window 는 그 E 고정 (given_E 조건).
  매 디노이징 step 에서 보상 R(x0_hat) 의 gradient 를 ε 에 반영한다 (classifier guidance: ε' = ε - scale·√(1-ᾱ)·∇R).
  gradient 는 network 를 거쳐 S·A 와 E 모두에 전달된다. scale 0 = guidance 없음 (비교 기준).
실행: 생성한 E 로 MJX 모델을 만들고 (pd_replay.build_model), 생성한 target_q 를 50 Hz 로 그대로 넣는다 (재계획 없음).
  초기 state = 데이터의 실제 state (--s0-motion 의 primitive 시작), 몸통 높이만 E 의 발이 바닥에 닿게 맞춘다.
보상 = task_reward.TrackReward (TT 예측 궤적·MJX 실제 궤적에 같은 class).

usage (codesign/ 에서):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false ../transformer-transformer/.venv/bin/python \
      code/tt_dynamics/generate_task.py --out runs/pipeline_v1/task/track_try1
"""

import argparse
import imageio
import json
import pickle
import sys
from pathlib import Path

CODESIGN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODESIGN.parent / "transformer-transformer"))
sys.path.insert(0, str(CODESIGN / "code" / "motion_tracking"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mujoco import mjx  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pd_replay import build_model  # noqa: E402
from rollout_data import E_DIM, S_DIM, X_DIM, features  # noqa: E402
from t2.mjx_env.base_env import make_data, step  # noqa: E402
from task_reward import TrackReward, ZigzagReward  # noqa: E402
from track_env import TrackMjxEnv  # noqa: E402
from tt_model import Net, schedule, unbase  # noqa: E402

EP, FOOT_R, A = 500, 0.02, slice(S_DIM, X_DIM)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", type=Path, default=Path("runs/pipeline_v1/tt/h32_try4"))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--candidates", type=int, default=8)
parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 5.0], help="guidance 세기 (0 = 없음)")
parser.add_argument("--s0-motion", default="walkForwardSlow__r00")
parser.add_argument("--task", choices=("circle", "zigzag"), default="circle")
parser.add_argument("--radius", type=float, default=1.0, help="circle")
parser.add_argument("--half-width", type=float, default=0.3, help="트랙 반폭 (m)")
parser.add_argument("--amp", type=float, default=0.3, help="zigzag 진폭 (m)")
parser.add_argument("--period", type=float, default=2.0, help="zigzag 주기 (m)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
dev = "cuda"
args.out.mkdir(parents=True, exist_ok=True)
reward = (TrackReward(radius=args.radius, half_width=args.half_width) if args.task == "circle"
          else ZigzagReward(amp=args.amp, period=args.period, half_width=args.half_width))

# ---- TT ----
ck = torch.load(args.ckpt / "checkpoint.pt", weights_only=False)
ta = ck["args"]
H = ta["horizon"]
net = Net(H, ta["d_model"], ta["layers"]).to(dev)
net.load_state_dict(ck["net"])
net.eval().requires_grad_(False)
mu, sd = torch.tensor(ck["mu"], device=dev), torch.tensor(ck["sd"], device=dev)
emu, esd = (0.5, 0.5) if ta["norm"] == "minmax" else (0.5, 0.25)
beta, abar = schedule(ta["diff_steps"], dev)
CLIP = ta["clip"]

# ---- s0: 데이터의 실제 state ----
z = np.load(CODESIGN / "runs/pipeline_v1/rollouts/e000/primitive_seed0.npz")
b0 = int(np.where(z["motions"][z["clip"][:, 0]] == args.s0_motion)[0][0])
qpos0, qvel0 = z["qpos"][b0, 0], z["qvel"][b0, 0]
S0 = torch.tensor(features(z["qpos"][b0, :1], z["qvel"][b0, :1], z["contact"][b0, :1])[0], device=dev)


def ddim(x0, e0, kx, ke, pad, cur, m, scale):
    """window 하나를 guidance 와 함께 DDIM 으로 생성. 반환 = 정규화 공간 x [B,H,X], e [B,13]."""
    x, e = torch.randn_like(x0), torch.randn_like(e0)
    for k in reversed(range(len(beta))):
        ab = abar[k]
        ab_prev = abar[k - 1] if k > 0 else torch.ones((), device=dev)
        kk = torch.full((len(x),), k, device=dev, dtype=torch.float32)
        with torch.enable_grad():
            xr, er = x.detach().requires_grad_(scale > 0), e.detach().requires_grad_(scale > 0)
            ex, ee = net(torch.where(kx > 0, x0, xr), torch.where(ke > 0, e0, er), kx, ke, kk, pad)
            if scale > 0:
                x0h = torch.where(kx > 0, x0, ((xr - (1 - ab).sqrt() * ex) / ab.sqrt()).clamp(-CLIP, CLIP))
                R, _ = reward(unbase(x0h * sd + mu, cur), m)
                gx, ge = torch.autograd.grad(R.sum(), [xr, er])
                ex, ee = ex - scale * (1 - ab).sqrt() * gx, ee - scale * (1 - ab).sqrt() * ge
        ex, ee = ex.detach(), ee.detach()
        x0h = ((x - (1 - ab).sqrt() * ex) / ab.sqrt()).clamp(-CLIP, CLIP)
        e0h = ((e - (1 - ab).sqrt() * ee) / ab.sqrt()).clamp(-CLIP, CLIP)
        epx, epe = (x - ab.sqrt() * x0h) / (1 - ab).sqrt(), (e - ab.sqrt() * e0h) / (1 - ab).sqrt()
        x, e = ab_prev.sqrt() * x0h + (1 - ab_prev).sqrt() * epx, ab_prev.sqrt() * e0h + (1 - ab_prev).sqrt() * epe
    return torch.where(kx > 0, x0, x), torch.where(ke > 0, e0, e)


@torch.no_grad()
def generate(scale, n, seed):
    torch.manual_seed(seed)
    traj = torch.zeros(n, EP, X_DIM, device=dev)
    traj[:, 0, :S_DIM] = S0
    cur = traj[:, 0].clone()
    e_norm = None
    t0 = 0
    while t0 < EP - 1:
        w = min(H, EP - t0)
        m = (torch.arange(H, device=dev) < w).expand(n, H)
        x0 = torch.zeros(n, H, X_DIM, device=dev)
        s = cur.clone()
        s[:, 0:2], s[:, 3], s[:, 4] = 0, 1, 0  # window 기준 원점
        x0[:, 0] = (s - mu) / sd
        kx = torch.zeros_like(x0)
        kx[:, 0, :S_DIM] = 1
        kx = kx * m[..., None]
        ke = torch.ones(n, E_DIM, device=dev) * (e_norm is not None)
        e0 = e_norm if e_norm is not None else torch.zeros(n, E_DIM, device=dev)
        xw, ew = ddim(x0 * m[..., None], e0, kx, ke, ~m, cur, m, scale)
        if e_norm is None:
            e_norm = ew
        ep = unbase(xw * sd + mu, cur)
        traj[:, t0 + 1:t0 + w, :S_DIM] = ep[:, 1:w, :S_DIM]
        traj[:, t0:t0 + w, A] = ep[:, :w, A]
        cur = ep[:, w - 1].clone()
        t0 += w - 1
    return traj, (e_norm * esd + emu).clamp(0, 1)


# ---- MJX 실행 ----
expert = CODESIGN / "runs/pipeline_v1/experts" / args.s0_motion
cfg = pickle.load(open(sorted(expert.glob("wandb/offline-run-*/files/config.pkl"))[-1], "rb"))
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
choice = cfg.get("choice")
ecfg["reference_dir"] = str(CODESIGN / "runs/pipeline_v1/retargeting/e000" / choice)  # 보상 계산용 reference (실행에는 무관)
ecfg["clips"] = [args.s0_motion]
jit_step = None


def execute(E_np, tq):
    """E [13], target_q [EP,12] -> MJX 궤적 토큰 [EP,X], mask [EP], 낙상 step (없으면 -1)."""
    global jit_step
    env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                                     torque_limit=cfg.get("torque_limit"), uniforms=E_np), **ecfg)
    if jit_step is None:
        jit_step = jax.jit(lambda model, d, c: step(model, d, c, env.n_substeps))
    q = qpos0.copy()
    d = mujoco.MjData(env.mj_model)
    d.qpos[:] = q
    mujoco.mj_kinematics(env.mj_model, d)
    q[2] -= d.geom_xpos[np.asarray(env._foot_ids), 2].min() - FOOT_R  # 발이 바닥에 닿게
    data = mjx.forward(env.mjx_model, make_data(env.mj_model, qpos=jnp.asarray(q), qvel=jnp.asarray(qvel0), impl=env.impl,
                                               nconmax=env._nconmax, njmax=env._njmax))
    off, sc, (lo, hi) = np.asarray(env.action_offset), float(env.action_scale), env.clip_actions
    feet = np.asarray(env._foot_ids)
    qs, vs, cs = [], [], []
    fell = -1
    for t in range(EP):
        qs.append(np.asarray(data.qpos)), vs.append(np.asarray(data.qvel))
        geom, dist = np.asarray(data._impl.contact.geom), np.asarray(data._impl.contact.dist)
        cs.append([((geom == f).any(1) & (dist < 1e-3)).any() for f in feet])
        w, x, y, _ = qs[-1][3:7]
        if 1 - 2 * (x * x + y * y) < env.termination_fellover_threshold:  # 몸통 z 축의 world z 성분
            fell = t
            break
        if t == EP - 1:
            break
        a = np.clip((tq[t] - off[:12]) / sc, lo, hi)
        ctrl = off.copy()
        ctrl[:12] = off[:12] + a * sc
        data = jit_step(env.mjx_model, data, jnp.asarray(ctrl))
    L = len(qs)
    x = np.zeros((EP, X_DIM), np.float32)
    x[:L, :S_DIM] = features(np.array(qs), np.array(vs), np.array(cs))
    x[:L, A] = tq[:L]
    m = np.zeros(EP, bool)
    m[:L] = True
    return x, m, fell, np.array(qs), env.mj_model


def score(x, m):
    total, t = reward(torch.as_tensor(x), torch.as_tensor(m))
    return total.numpy(), {k: v.numpy() for k, v in t.items()}


(args.out / "designs").mkdir(exist_ok=True)
(args.out / "videos").mkdir(exist_ok=True)


def save_design(model, E_np, name):
    """생성 형상: json (build_model 로 재생성 가능한 정의) + 컴파일된 MuJoCo 모델 .mjb."""
    json.dump(dict(choice=choice, uniforms=E_np.round(6).tolist(), kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                   torque_limit=cfg.get("torque_limit"), total_mass_kg=float(model.body_subtreemass[0]),
                   rebuild="pd_replay.build_model(choice, kp=kp, tau_scale=tau_scale, torque_limit=torque_limit, uniforms=uniforms)"),
              open(args.out / "designs" / f"{name}.json", "w"), indent=1)
    mujoco.mj_saveModel(model, str(args.out / "designs" / f"{name}.mjb"), None)


def render(model, qraw, pred_i, path, size=(640, 368)):
    """MJX 실제 궤적 재생 영상. 바닥 점: 흰색 = 트랙 경계 (task 기준선), 파랑 = TT 예측 경로. 카메라는 몸통을 따라간다."""
    model.vis.global_.offwidth, model.vis.global_.offheight = size
    r = mujoco.Renderer(model, size[1], size[0])
    d = mujoco.MjData(model)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1  # TT 형상 geom
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 2.2, 135.0, -35.0
    # s0 기준 좌표 -> world (출발 xy·yaw)
    w, x, y, z_ = qraw[0, 3:7]
    yaw0 = np.arctan2(2 * (w * z_ + x * y), 1 - 2 * (y * y + z_ * z_))
    c0, s0_ = np.cos(yaw0), np.sin(yaw0)
    to_world = lambda p: qraw[0, :2] + np.stack([c0 * p[:, 0] - s0_ * p[:, 1], s0_ * p[:, 0] + c0 * p[:, 1]], 1)  # noqa: E731
    marks = [(to_world(pts[::4]), (1, 1, 1, 1)) for pts, center in reward.guide_lines() if not center]
    marks.append((to_world(pred_i[::10, :2]), (0.2, 0.4, 1, 1)))
    frames = []
    for q in qraw:
        d.qpos[:] = q
        mujoco.mj_forward(model, d)
        cam.lookat[:] = q[:3]
        r.update_scene(d, camera=cam, scene_option=opt)
        for pts, rgba in marks:
            for p in pts:
                if r.scene.ngeom >= r.scene.maxgeom:
                    break
                g = r.scene.geoms[r.scene.ngeom]
                mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.02, 0, 0]), np.array([p[0], p[1], 0.01]),
                                    np.eye(3).flatten(), np.array(rgba, np.float32))
                r.scene.ngeom += 1
        frames.append(r.render())
    imageio.mimsave(path, frames, fps=50)
    r.close()


summary = dict(reward=reward.describe(), task=args.task, ckpt=str(args.ckpt), s0_motion=args.s0_motion, candidates=args.candidates, runs={})
fig, axes = plt.subplots(1, len(args.scales), figsize=(5 * len(args.scales), 5), squeeze=False)
for si, scale in enumerate(args.scales):
    traj, E = generate(scale, args.candidates, args.seed)
    pred = traj.cpu().numpy()
    pm = np.ones((args.candidates, EP), bool)
    p_total, p_terms = score(pred, pm)
    rows = []
    mj = np.zeros_like(pred)
    mm = np.zeros_like(pm)
    for i in range(args.candidates):
        mj[i], mm[i], fell, qraw, model = execute(E[i].cpu().numpy(), pred[i, :, A])
        name = f"scale_{scale:g}_cand{i}"
        save_design(model, E[i].cpu().numpy(), name)
        render(model, qraw, pred[i], args.out / "videos" / f"{name}.mp4")
        rows.append(fell)
    r_total, r_terms = score(mj, mm)
    L = mm.sum(1)
    path_err = [float(np.linalg.norm(pred[i, :L[i], :2] - mj[i, :L[i], :2], axis=-1).mean()) for i in range(args.candidates)]
    res = []
    for i in range(args.candidates):
        res.append(dict(E=E[i].cpu().numpy().round(3).tolist(),
                        pred_total=float(p_total[i]), pred_progress=float(p_terms["r_progress"][i] * reward.unit_scale),
                        mjx_total=float(r_total[i]), mjx_progress=float(r_terms["r_progress"][i] * reward.unit_scale),
                        mjx_fell_step=int(rows[i]), mjx_valid_s=float(L[i] * 0.02), path_err_m=path_err[i],
                        pred_terms={k: float(v[i]) for k, v in p_terms.items()}, mjx_terms={k: float(v[i]) for k, v in r_terms.items()}))
        print(f"scale {scale:g} #{i}: TT 예측 {res[-1]['pred_progress']:.2f} {reward.unit} (R {p_total[i]:.2f}) | MJX {res[-1]['mjx_progress']:.2f} {reward.unit} "
              f"(R {r_total[i]:.2f}), 낙상 {rows[i]}, 경로 차 {path_err[i]:.2f} m | E[:3] {res[-1]['E'][:3]}", flush=True)
    summary["runs"][f"scale_{scale:g}"] = res
    np.savez_compressed(args.out / f"scale_{scale:g}.npz", pred=pred, mjx=mj, mjx_mask=mm, E=E.cpu().numpy())
    ax = axes[0, si]
    for pts, center in reward.guide_lines():
        ax.plot(pts[:, 0], pts[:, 1], "k-" if center else "k:", lw=1 if center else 0.7)
    for i in range(args.candidates):
        c = plt.cm.tab10(i % 10)
        ax.plot(pred[i, :, 0], pred[i, :, 1], "--", color=c, lw=1)
        ax.plot(mj[i, :L[i], 0], mj[i, :L[i], 1], "-", color=c, lw=1.5)
        if rows[i] >= 0:
            ax.plot(mj[i, L[i] - 1, 0], mj[i, L[i] - 1, 1], "x", color=c, ms=8)
    ax.set_aspect("equal")
    ax.set_title(f"guidance {scale:g} (dashed: TT, solid: MJX, x: fall)")
json.dump(summary, open(args.out / "summary.json", "w"), indent=1, ensure_ascii=False)
fig.tight_layout()
fig.savefig(args.out / "paths.png", dpi=110)
print("->", args.out)
