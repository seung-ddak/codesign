"""7 단계: 학습한 TT 로 복합 task 의 형상 E·state S·action A 를 reward guidance 로 생성하고, E·A 를 MJX 에서 open-loop 실행한다.

생성: window (H step) 를 이어 붙여 10 s (500 step). 첫 window 는 E 도 노이즈에서 생성 (joint 조건), 이후 window 는 그 E 고정 (given_E 조건).
  --free-s0: 첫 window 를 free 조건 (mode 3) 으로 s0 까지 노이즈에서 생성한다 (TT 를 --modes 0 1 2 3 으로 학습해야 한다).
  --anchor-s0 (old alias --rest-s0): 첫 window Mode 4. 시작 위치·Yaw 좌표 기준만 고정. 속도·접촉·높이·관절각은 MJX 데이터 분포에서 E 와 함께 생성.
  매 디노이징 step 에서 보상 R(x0_hat) 의 gradient 를 ε 에 반영한다 (classifier guidance: ε' = ε - scale·√(1-ᾱ)·∇R).
  gradient 는 network 를 거쳐 S·A 와 E 모두에 전달된다. scale 0 = guidance 없음 (비교 기준).
실행: 생성한 E 로 MJX 모델을 만들고 (pd_replay.build_model), 생성한 target_q 를 50 Hz 로 그대로 넣는다 (재계획 없음).
  초기 state = 데이터의 실제 state (--s0-motion 의 primitive 시작), --free-s0 이면 생성한 s0. 몸통 높이만 E 의 발이 바닥에 닿게 맞춘다.
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
from shared_e_guidance import optimize_shared_e  # noqa: E402
from joint_diffusion_guidance import joint_diffusion_full_reward  # noqa: E402
from mujoco import mjx  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pd_replay import build_model  # noqa: E402
from rollout_data import E_DIM, S_DIM, X_DIM, features  # noqa: E402
from s0_geometry import s0_to_q  # noqa: E402
from initial_geometry_gate import bounded_lift, classify_initial_support  # noqa: E402
from t2.mjx_env.base_env import make_data, step  # noqa: E402
from obstacle_task import ObstacleReward, add_pole, envelope_radius, pole_contact
from task_reward import StraightReward, TrackReward, ZigzagReward  # noqa: E402
from track_env import TrackMjxEnv  # noqa: E402
from tt_model import Net, check_fixed_e_rest_checkpoint, check_free_checkpoint, check_rest_checkpoint, condition_masks, rest_s0, schedule, unbase  # noqa: E402

EP, FOOT_R, A = 500, 0.02, slice(S_DIM, X_DIM)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", type=Path, default=Path("runs/pipeline_v1/tt/h32_try4"))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--candidates", type=int, default=8)
parser.add_argument("--scales", type=float, nargs="+", default=[0.0, 1.0, 5.0], help="guidance 세기 (0 = 없음)")
parser.add_argument("--s0-motion", default="walkForwardSlow__r00")
parser.add_argument("--expert-config", type=Path, help="expert config directory (used to instantiate MJX controller; defaults to original expert)")
parser.add_argument("--task", choices=("circle", "zigzag", "straight", "obstacle"), default="circle")
parser.add_argument("--obstacle-count", type=int, default=1, help="upright poles at 1m intervals")
parser.add_argument("--radius", type=float, default=1.0, help="circle")
parser.add_argument("--half-width", type=float, default=None, help="corridor half-width (m); task-specific defaults when omitted")
parser.add_argument("--amp", type=float, default=0.3, help="zigzag 진폭 (m)")
parser.add_argument("--period", type=float, default=2.0, help="zigzag 주기 (m)")
parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--free-s0", action="store_true", help="s0 도 생성 (free 조건)")
parser.add_argument("--anchor-s0", "--rest-s0", dest="rest_s0", action="store_true", help="Mode 4: anchor XY=(0,0), yaw=0; learn remaining S0 and E (old alias --rest-s0)")
parser.add_argument("--s-gradient-scale", type=float, default=1.0, help="S-token guidance multiplier; E and A gradients unchanged")
parser.add_argument("--no-guide-s", action="store_true", help="S 토큰 guidance 끔 (E·A 만). S 는 TT dynamics 로만 따라온다")
parser.add_argument("--no-s0-align", action="store_true", help="execute generated S0 exactly; do not shift body height")
parser.add_argument("--geometry-s0", action="store_true",
                    help="opt-in bounded post-denoising S0 floor-geometry projection; preserve raw TT prediction")
parser.add_argument("--max-geometry-lift-m", type=float, default=0.04,
                    help="maximum bounded vertical correction of generated S0 (metres)")
parser.add_argument("--allow-airborne-s0", action="store_true",
                    help="test geometry-clear zero-contact starts in MJX instead of rejecting as unsupported")
parser.add_argument("--no-render", action="store_true")
parser.add_argument("--joint-full-guidance", action="store_true",
                    help="per-denoising-step reward gradient guides noisy E and complete 10s S/A (not SPSA)")
parser.add_argument("--e-rest-s0", action="store_true",
                    help="with --joint-full-guidance: S0 = home stance of the current E (feet on floor), not generated")
parser.add_argument("--fixed-e", nargs="+", default=None, metavar="DESIGN",
                    help="co-design control: E = these training designs (designs.json), --candidates noise seeds each, "
                         "shared across designs. S0 = rest_s0_of(E), every window Mode 2 + same guidance")
parser.add_argument("--replan", type=int, default=0, metavar="K",
                    help="every K MJX steps, regenerate the next window from the ACTUAL MJX state "
                         "(Mode 2, E fixed, same guidance). 0 = open-loop (default)")
parser.add_argument("--expert-warmup-s0", type=float, default=0.0, metavar="SEC",
                    help="with --e-rest-s0: expert walks closed-loop on the current E for up to SEC s; S0 = first "
                         "balanced state after 0.5 s (same process as TT training data, so S0 is in-distribution)")
parser.add_argument("--shared-e-guidance", action="store_true",
                    help="estimate full-horizon reward gradient for one E shared across TT windows (SPSA)")
parser.add_argument("--shared-e-iters", type=int, default=4, help="SPSA outer iterations per guidance scale")
parser.add_argument("--shared-e-eval-baseline", action="store_true",
                    help="also MJX-evaluate same-scale, same-noise pre-optimization E and action (extra compute)")
parser.add_argument("--shared-e-perturb", type=float, default=0.04, help="morphology finite-difference probe (E in [0,1])")
parser.add_argument("--shared-e-step-size", type=float, default=0.08, help="maximum normalized morphology update")
args = parser.parse_args()
EP = round(args.seconds / 0.02)
if EP < 2:
    parser.error("--seconds must contain at least two 50 Hz states")
if args.obstacle_count < 1:
    parser.error("--obstacle-count must be positive")
if not np.isfinite(args.s_gradient_scale) or args.s_gradient_scale < 0:
    parser.error("--s-gradient-scale must be finite and nonnegative")
if args.s_gradient_scale != 1 and not args.joint_full_guidance:
    parser.error("--s-gradient-scale requires --joint-full-guidance")
dev = args.device
if args.task == "obstacle" and (args.out / "summary.json").exists():
    parser.error("existing obstacle result preserved; choose a new --out directory")
args.out.mkdir(parents=True, exist_ok=True)
if args.half_width is None:
    args.half_width = 0.15 if args.task in ("zigzag", "straight") else 0.3
if args.task == "obstacle" and (not args.e_rest_s0 or args.fixed_e or args.replan or args.expert_warmup_s0):
    parser.error("obstacle requires joint --e-rest-s0, without fixed E, warmup or replanning")
reward = (TrackReward(radius=args.radius, half_width=args.half_width) if args.task == "circle"
          else ZigzagReward(amp=args.amp, period=args.period, half_width=args.half_width) if args.task == "zigzag"
          else ObstacleReward(count=args.obstacle_count) if args.task == "obstacle" else StraightReward(half_width=args.half_width))

# ---- TT ----
ck = torch.load(args.ckpt / "checkpoint.pt", weights_only=False, map_location=dev)
ta = ck["args"]
if args.free_s0:
    check_free_checkpoint(ta)
if args.rest_s0:
    check_rest_checkpoint(ta)
assert not (args.free_s0 and args.rest_s0)
if args.shared_e_guidance and args.joint_full_guidance:
    parser.error("select one of --shared-e-guidance (SPSA) and --joint-full-guidance (joint diffusion)")
if args.joint_full_guidance:
    if not args.rest_s0 or args.free_s0:
        parser.error("--joint-full-guidance requires --rest-s0")
    if 2 not in ta.get("modes", []):
        parser.error("--joint-full-guidance requires checkpoint trained with Mode 2")
if args.e_rest_s0 and (not args.joint_full_guidance or args.geometry_s0):
    parser.error("--e-rest-s0 requires --joint-full-guidance and no --geometry-s0 (S0 already touches the floor)")
if args.expert_warmup_s0 > 0 and not args.e_rest_s0:
    parser.error("--expert-warmup-s0 requires --e-rest-s0")
if args.shared_e_guidance:
    if not args.rest_s0 or args.free_s0:
        parser.error("--shared-e-guidance currently requires --rest-s0 (mode 4 initial E, mode 5 fixed-E rest S0)")
    check_fixed_e_rest_checkpoint(ta)
    if 2 not in ta.get("modes", []):
        parser.error("--shared-e-guidance requires Mode 2 for subsequent windows")
    if args.shared_e_iters < 0 or args.shared_e_perturb <= 0 or args.shared_e_step_size <= 0:
        parser.error("invalid shared-E optimizer parameters")
GEN_S0 = args.free_s0 or args.rest_s0
if args.geometry_s0 and (not GEN_S0 or not args.no_s0_align):
    parser.error('--geometry-s0 requires generated S0 and --no-s0-align; legacy alignment must stay disabled')
if args.max_geometry_lift_m <= 0 or not np.isfinite(args.max_geometry_lift_m):
    parser.error('--max-geometry-lift-m must be positive and finite')
H = ta["horizon"]
if not 0 <= args.replan < H:
    parser.error(f"--replan K must satisfy 0 <= K < horizon ({H})")
PER_DESIGN = args.candidates
if args.fixed_e:
    args.candidates *= len(args.fixed_e)  # 후보 = 형상 × seed (형상 순서대로 PER_DESIGN 개씩)
net = Net(H, ta["d_model"], ta["layers"]).to(dev)
net.load_state_dict(ck["net"])
net.eval().requires_grad_(False)
mu, sd = torch.tensor(ck["mu"], device=dev), torch.tensor(ck["sd"], device=dev)
emu, esd = (0.5, 0.5) if ta["norm"] == "minmax" else (0.5, 0.25)
beta, abar = schedule(ta["diff_steps"], dev)
CLIP = ta["clip"]

# ---- S0: legacy primitive start for fixed-S0 mode only ----
# Rest-S0 samples its own morphology-dependent start; it must not require an
# unrelated original e000 rollout file to exist on the evaluation machine.
if GEN_S0:
    qpos0 = qvel0 = None
    S0 = torch.zeros(S_DIM, device=dev)
else:
    with np.load(CODESIGN / "runs/pipeline_v1/rollouts/e000/primitive_seed0.npz") as z:
        b0 = int(np.where(z["motions"][z["clip"][:, 0]] == args.s0_motion)[0][0])
        qpos0, qvel0 = z["qpos"][b0, 0], z["qvel"][b0, 0]
        S0 = torch.tensor(features(z["qpos"][b0, :1], z["qvel"][b0, :1], z["contact"][b0, :1])[0], device=dev)



if not GEN_S0:
    _q, _v = s0_to_q(S0.cpu().numpy())
    assert np.allclose(features(_q[None], _v[None], np.zeros((1, 4)))[0, 2:38],
                       S0.cpu().numpy()[2:38], atol=1e-4)


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
                # When E is conditioned (modes 1/2/5), er is unused by the
                # network. Do not fail just because E is fixed in this window.
                gx, ge = torch.autograd.grad(R.sum(), [xr, er], allow_unused=True)
                if gx is None:
                    gx = torch.zeros_like(xr)
                if ge is None:
                    ge = torch.zeros_like(er)
                if args.no_guide_s:
                    gx[..., :S_DIM] = 0
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
    if args.free_s0:
        traj[:, 0, :S_DIM] = 0
        traj[:, 0, 3] = 1  # 위치 0, yaw 0 (window 기준 원점). 나머지는 첫 window 에서 생성
    if args.rest_s0:
        traj[:, 0, :S_DIM] = rest_s0().to(dev)  # Only XY/yaw anchored; all other S0 features learned
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
        mode = 2 if e_norm is not None else (3 if args.free_s0 else 4 if args.rest_s0 else 0)
        kx, ke = condition_masks(torch.full((n,), mode, device=dev), m)
        e0 = e_norm if e_norm is not None else torch.zeros(n, E_DIM, device=dev)
        xw, ew = ddim(x0 * m[..., None], e0, kx, ke, ~m, cur, m, scale)
        ep = unbase(xw * sd + mu, cur)
        if e_norm is None:
            e_norm = ew
            traj[:, 0, 5:S_DIM] = ep[:, 0, 5:S_DIM]  # free-s0: 생성한 s0 (위치·yaw 는 원점 유지). 아니면 S0 그대로
            traj[:, 0, 2] = ep[:, 0, 2]
        traj[:, t0 + 1:t0 + w, :S_DIM] = ep[:, 1:w, :S_DIM]
        traj[:, t0:t0 + w, A] = ep[:, :w, A]
        cur = ep[:, w - 1].clone()
        t0 += w - 1
    return traj, (e_norm * esd + emu).clamp(0, 1)


def plan_window(cur, e_norm, scale, seed, w):
    """한 window 재생성: E 고정 + S0 = cur (episode frame, MJX 실제 state) 인 Mode 2. generate() 의 window 블록과 같다."""
    torch.manual_seed(seed)
    n = len(cur)
    m = (torch.arange(H, device=dev) < w).expand(n, H)
    x0 = torch.zeros(n, H, X_DIM, device=dev)
    s = cur.clone()
    s[:, 0:2], s[:, 3], s[:, 4] = 0, 1, 0  # window 기준 원점
    x0[:, 0] = (s - mu) / sd
    kx, ke = condition_masks(torch.full((n,), 2, device=dev), m)
    xw, _ = ddim(x0 * m[..., None], e_norm, kx, ke, ~m, cur, m, scale)
    return unbase(xw * sd + mu, cur)[:, :w].detach()


def generate_fixed_designs(scale):
    """co-design 대조군: E = 학습 형상 고정, S0 = rest_s0_of(E), window 마다 plan_window (Mode 2 + 같은 guidance).
    형상마다 같은 seed 집합 (후보 j 의 노이즈 공유) 이라 형상 간 차이는 E 에서만 온다."""
    designs = json.loads((CODESIGN / "runs/pipeline_v1/retargeting/designs.json").read_text())
    E = torch.tensor([designs[d] for d in args.fixed_e for _ in range(PER_DESIGN)], device=dev, dtype=torch.float32)
    e_norm = (E - emu) / esd
    traj = torch.zeros(len(E), EP, X_DIM, device=dev)
    traj[:, 0, :S_DIM] = rest_s0_of(E)
    cur, t0 = traj[:, 0].clone(), 0
    while t0 < EP - 1:
        w = min(H, EP - t0)
        ep = torch.cat([plan_window(cur[b:b + 1], e_norm[b:b + 1], scale, args.seed + 7919 * (b % PER_DESIGN) + t0, w)
                        for b in range(len(E))])
        traj[:, t0 + 1:t0 + w, :S_DIM] = ep[:, 1:w, :S_DIM]
        traj[:, t0:t0 + w, A] = ep[:, :w, A]
        cur, t0 = ep[:, w - 1].clone(), t0 + w - 1
    return traj, E


@torch.no_grad()
def sample_rest_morphology(n, seed):
    """Draw E from learned Mode 4 prior once, before 10s task optimization."""
    torch.manual_seed(seed)
    cur = torch.zeros(n, X_DIM, device=dev)
    cur[:, :S_DIM] = rest_s0().to(device=dev)
    m = (torch.arange(H, device=dev) < min(H, EP)).expand(n, H)
    x0 = torch.zeros(n, H, X_DIM, device=dev)
    x0[:, 0] = (cur - mu) / sd
    kx, ke = condition_masks(torch.full((n,), 4, dtype=torch.long, device=dev), m)
    _, e = ddim(x0 * m[..., None], torch.zeros(n, E_DIM, device=dev),
                kx, ke, ~m, cur, m, 0.0)
    physical = (e * esd + emu).clamp(0.05, 0.95)
    physical[:, [0, 11, 12]] = 0.5  # inert under current arm-free A1 topology
    return physical


@torch.no_grad()
def generate_fixed_e(E_physical, scale, seed):
    """Roll out the entire horizon under one FIXED E.

    First window uses Mode 5: E known, only initial XY/yaw anchored. All later
    windows use Mode 2: E and the generated preceding-window S0 known.
    Same seed provides common diffusion noise across finite-difference
    morphology probes. E is never re-sampled between windows.
    """
    torch.manual_seed(seed)
    n = len(E_physical)
    e0 = (E_physical - emu) / esd
    traj = torch.zeros(n, EP, X_DIM, device=dev)
    traj[:, 0, :S_DIM] = rest_s0().to(device=dev)
    cur = traj[:, 0].clone()
    t0 = 0
    while t0 < EP - 1:
        w = min(H, EP - t0)
        m = (torch.arange(H, device=dev) < w).expand(n, H)
        x0 = torch.zeros(n, H, X_DIM, device=dev)
        s = cur.clone()
        s[:, 0:2], s[:, 3], s[:, 4] = 0, 1, 0
        x0[:, 0] = (s - mu) / sd
        mode = 5 if t0 == 0 else 2
        kx, ke = condition_masks(torch.full((n,), mode, device=dev), m)
        xw, _ = ddim(x0 * m[..., None], e0, kx, ke, ~m, cur, m, scale)
        ep = unbase(xw * sd + mu, cur)
        if t0 == 0:
            traj[:, 0, 5:S_DIM] = ep[:, 0, 5:S_DIM]
            traj[:, 0, 2] = ep[:, 0, 2]
        traj[:, t0 + 1:t0 + w, :S_DIM] = ep[:, 1:w, :S_DIM]
        traj[:, t0:t0 + w, A] = ep[:, :w, A]
        cur = ep[:, w - 1].clone()
        t0 += w - 1
    return traj


# ---- MJX 실행 ----
expert = args.expert_config if args.expert_config is not None else CODESIGN / "runs/pipeline_v1/experts" / args.s0_motion
cfg = pickle.load(open(sorted(expert.glob("wandb/offline-run-*/files/config.pkl"))[-1], "rb"))
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
if dev == "cpu":
    ecfg["impl"] = "jax"
choice = cfg.get("choice")
ecfg["reference_dir"] = str(CODESIGN / "runs/pipeline_v1/retargeting/e000" / choice)  # 보상 계산용 reference (실행에는 무관)
ecfg["clips"] = [args.s0_motion]
jit_step = None


def rest_s0_of(E_batch, settle_s=1.0):
    """S0 = g(E): home 자세 PD 목표로 settle_s 초 세워 둔 안착 상태 (실행과 같은 접촉 설정). E 가 바뀌면 S0 도 바뀐다."""
    out = []
    radii = []
    for e in E_batch.detach().cpu().numpy().astype(np.float64):
        env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                                         torque_limit=cfg.get("torque_limit"), uniforms=e), **ecfg)
        m = env.mj_model
        if args.task == "obstacle":
            radii.append(envelope_radius(m))
        d = mujoco.MjData(m)
        d.qpos[:] = m.keyframe("home").qpos
        d.qpos[2] = 1.0
        mujoco.mj_forward(m, d)
        feet = np.asarray(env._foot_ids)
        d.qpos[2] -= float((d.geom_xpos[feet, 2] - m.geom_size[feet, 0]).min())
        d.ctrl[:] = np.asarray(env.action_offset)
        for _ in range(round(settle_s / m.opt.timestep)):
            mujoco.mj_step(m, d)
        floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        contact = [any(floor in (c.geom1, c.geom2) and f in (c.geom1, c.geom2) and c.dist <= 1e-3
                       for c in d.contact[:d.ncon]) for f in feet]
        out.append(features(d.qpos[None].copy(), d.qvel[None].copy(), np.array([contact]))[0])
    if args.task == "obstacle":
        # The settled CPU geometry is not differentiable; E still guides through TT dynamics.
        reward.robot_radius = radii
    return torch.tensor(np.stack(out), device=dev, dtype=torch.float32)


_warm = {}


def warm_s0_of(E_batch):
    """S0 = g(E): expert (multi-design, E 를 obs 로 받음) 를 E 위에서 closed-loop 로 걸린 뒤 상태.
    TT 학습 데이터와 같은 생성 방식이라 분포 안. reference 는 E 가 가장 가까운 학습 형상의 retarget 을 쓴다.
    env.step 은 base env 의 mjx_model·reference 속성을 trace 시점에 바꿔 끼워 한 번만 compile (multi_design.py 와 같은 방식)."""
    from multi_design import REFERENCE_ATTRS
    from policy_io import load_run
    retarget = CODESIGN / "runs/pipeline_v1/retargeting"
    mk = lambda u, name: TrackMjxEnv(  # noqa: E731
        mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                        torque_limit=cfg.get("torque_limit"), uniforms=u),
        **{**ecfg, "reference_dir": str(retarget / name / choice)})
    if not _warm:
        designs = json.loads((retarget / "designs.json").read_text())
        names = list(cfg.train_designs)
        ck = sorted(expert.glob("wandb/offline-run-*/files/checkpoints/*"), key=lambda p: int(p.name))[-1]
        _, policy = load_run(ck)
        base = mk(designs[names[0]], names[0])
        base.mjx_model = base.mjx_model.replace(geom_sameframe=np.zeros_like(base.mjx_model.geom_sameframe))
        feet, floor = jnp.asarray(base._foot_ids), int(base.mj_model.geom("floor").id)
        foot_r = jnp.asarray(base.mj_model.geom_size[base._foot_ids, 0])
        T = round(args.expert_warmup_s0 / base.dt)

        def run(model, refs, u, key):
            old = {k: getattr(base, k) for k in ("mjx_model", *REFERENCE_ATTRS)}
            base.mjx_model = model
            for k in REFERENCE_ATTRS:
                setattr(base, k, refs[k])
            def snap(d):
                g, dist = d._impl.contact.geom, d._impl.contact.dist
                c = jnp.stack([((((g[:, 0] == f) & (g[:, 1] == floor)) | ((g[:, 1] == f) & (g[:, 0] == floor)))
                                & (dist <= 1e-3)).any() for f in feet])
                return d.qpos, d.qvel, c, (d.geom_xpos[feet, 2] - foot_r).min()

            def body(s, _):
                s = base.step(s, policy(jnp.concatenate((s.obs, u)), key)[0])
                return s, snap(s.data)

            try:
                s = base.reset_with_traj(key, 0, base.clip_start[0])
                _, traj = jax.lax.scan(body, s, None, length=T)
            finally:
                for k, v in old.items():
                    setattr(base, k, v)
            return traj

        _warm.update(base=base, run=jax.jit(run), names=names,
                     bank=np.array([designs[n] for n in names], np.float32))
    base, out = _warm["base"], []
    for e in E_batch.detach().cpu().numpy().astype(np.float32):
        near = _warm["names"][int(np.linalg.norm(_warm["bank"] - e, axis=1).argmin())]
        env = mk(e, near)
        # MJX compile-time metadata must match the base model (as in multi_design.make_bank).
        model = env.mjx_model.replace(stat=base.mjx_model.stat, geom_sameframe=np.zeros_like(env.mjx_model.geom_sameframe),
                                      _impl=env.mjx_model._impl.replace(
                                          geom_rbound_hfield=base.mjx_model._impl.geom_rbound_hfield))
        refs = {k: jnp.asarray(getattr(env, k)) for k in REFERENCE_ATTRS}
        Q, V, Cn, fz = map(np.asarray, _warm["run"](model, refs, jnp.asarray(e), jax.random.PRNGKey(args.seed)))
        # 균형 잡힌 첫 순간 (0.5 s 이후): execute() gate 와 같은 기준 (2발 이상 접지, 발 침투 ≤ 2 mm, 서 있음).
        up = 1 - 2 * (Q[:, 4] ** 2 + Q[:, 5] ** 2)
        ok = (Cn.sum(1) >= 2) & (fz >= -2e-3) & (up > base.termination_fellover_threshold)
        ok[:round(0.5 / base.dt)] = False
        t = int(ok.argmax()) if ok.any() else len(Q) - 1
        q, v, c = Q[t], V[t], Cn[t]
        print(f"  warmup S0: ref {near} | t {(t + 1) * base.dt:.2f}s {'balanced' if ok.any() else 'NONE (fell/unbalanced)'} | "
              f"body z {q[2]:.3f} | upright {up[t]:.2f} | speed {np.linalg.norm(v[:3]):.2f} m/s | contacts {int(c.sum())} | "
              f"foot z {fz[t] * 1000:.1f} mm", flush=True)
        out.append(features(np.asarray(q)[None], np.asarray(v)[None], np.asarray(c)[None])[0])
    return torch.tensor(np.stack(out), device=dev, dtype=torch.float32)


def execute(E_np, tq, s0, replan=None):
    """E [13], target_q [EP,12], s0 [42] (free-s0 일 때만 사용) -> MJX 궤적 토큰 [EP,X], mask [EP], 낙상 step (없으면 -1).
    replan(t, qpos, qvel, contact) -> target_q [w,12]: args.replan step 마다 실제 state 로 다음 window 를 받는다."""
    global jit_step
    def execution_model():
        model = build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                            torque_limit=cfg.get("torque_limit"), uniforms=E_np,
                            scene_fn=(lambda spec: add_pole(spec, reward)) if args.task == "obstacle" else None)
        return model
    env = TrackMjxEnv(mj_model_fn=execution_model, **ecfg)
    if jit_step is None:
        jit_step = jax.jit(lambda model, d, c: step(model, d, c, env.n_substeps))
    # Reject obviously invalid generated starts BEFORE forward dynamics.
    if GEN_S0:
        s0 = np.asarray(s0, dtype=np.float64)
        gnorm = float(np.linalg.norm(s0[5:8]))
        if (not np.isfinite(s0).all() or not np.isfinite(tq).all()
                or not np.isfinite(gnorm) or not 0.9 <= gnorm <= 1.1
                or s0[2] <= 0):
            return (np.zeros((EP, X_DIM), np.float32), np.zeros(EP, bool),
                    -2, np.zeros((0, 19), np.float64), env.mj_model,
                    {"physically_valid": False, "execution_skipped": True,
                     "skip_reason": "nonfinite_or_invalid_generated_s0"})
    q, qv = s0_to_q(s0) if GEN_S0 else (qpos0.copy(), qvel0)
    d = mujoco.MjData(env.mj_model)
    d.qpos[:] = q
    d.qvel[:] = qv
    mujoco.mj_forward(env.mj_model, d)
    feet = np.asarray(env._foot_ids)
    floor = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    raw_foot_z = d.geom_xpos[feet, 2] - env.mj_model.geom_size[feet, 0]
    raw_contact = np.array([any(floor in (c.geom1, c.geom2) and f in (c.geom1, c.geom2) and c.dist <= 1e-3
                                for c in d.contact[:d.ncon]) for f in feet])
    raw_nonfoot_pen = max([max(0.0, -float(c.dist)) for c in d.contact[:d.ncon]
                           if floor in (c.geom1, c.geom2) and
                           (c.geom2 if c.geom1 == floor else c.geom1) not in feet] or [0.0])
    raw_self_pen = max([max(0.0, -float(c.dist)) for c in d.contact[:d.ncon]
                        if floor not in (c.geom1, c.geom2)] or [0.0])
    qnorm = float(np.linalg.norm(q[3:7]))
    joint_bad = any(env.mj_model.jnt_limited[j] and
                    not (env.mj_model.jnt_range[j, 0] - 1e-3 <= q[int(env.mj_model.jnt_qposadr[j])]
                         <= env.mj_model.jnt_range[j, 1] + 1e-3)
                    for j in range(env.mj_model.njnt))
    # Snapshot unmodified TT output before any optional, explicitly bounded
    # geometry correction. `pred` NPZ and TT reward remain UNCHANGED.
    preprojection = dict(body_height_m=float(q[2]),
                         foot_clearance_m=raw_foot_z.tolist(),
                         max_foot_penetration_m=float(max(0., -raw_foot_z.min())),
                         nonfoot_penetration_m=float(raw_nonfoot_pen),
                         self_penetration_m=float(raw_self_pen),
                         cpu_foot_contacts=int(raw_contact.sum()))
    projection = dict(needed=False, allowed=True, requested_lift_m=0.,
                      applied_lift_m=0., reason="disabled", max_lift_m=float(args.max_geometry_lift_m))
    if GEN_S0 and args.geometry_s0:
        projection = bounded_lift(raw_foot_z, raw_nonfoot_pen,
                                  max_lift_m=args.max_geometry_lift_m)
        if projection["applied_lift_m"] > 0:
            q = q.copy()
            q[2] += projection["applied_lift_m"]
            d.qpos[:] = q
            mujoco.mj_forward(env.mj_model, d)
            # Geometry MUST be rechecked after the corrected q is set.
            raw_foot_z = d.geom_xpos[feet, 2] - env.mj_model.geom_size[feet, 0]
            raw_contact = np.array([any(
                floor in (c.geom1, c.geom2) and f in (c.geom1, c.geom2) and c.dist <= 1e-3
                for c in d.contact[:d.ncon]) for f in feet])
            raw_nonfoot_pen = max([max(0.0, -float(c.dist)) for c in d.contact[:d.ncon]
                                   if floor in (c.geom1, c.geom2) and
                                   (c.geom2 if c.geom1 == floor else c.geom1) not in feet] or [0.0])
            raw_self_pen = max([max(0.0, -float(c.dist)) for c in d.contact[:d.ncon]
                                if floor not in (c.geom1, c.geom2)] or [0.0])
    z_shift = 0.0  # legacy mode only; --geometry-s0 never uses this alignment
    if not args.no_s0_align:
        z_shift = float(-raw_foot_z.min())
        q[2] += z_shift
    data = mjx.forward(env.mjx_model, make_data(env.mj_model, qpos=jnp.asarray(q), qvel=jnp.asarray(qv), impl=env.impl,
                                               nconmax=env._nconmax, njmax=env._njmax))
    initial_geom, initial_dist = np.asarray(data._impl.contact.geom), np.asarray(data._impl.contact.dist)
    mjx_floor_contact = [bool(((((initial_geom[:, 0] == f) & (initial_geom[:, 1] == floor)) |
                                 ((initial_geom[:, 1] == f) & (initial_geom[:, 0] == floor))) &
                                (initial_dist <= 1e-3)).any()) for f in feet]
    # Mode 4 now generates a moving initial state from the expert distribution.
    # Requiring all four feet on the floor would recreate the rare-condition
    # mismatch. Demand >=2 supporting contacts and no deep penetration instead.
    # This is a feasibility heuristic, NOT a guarantee of stability.
    support = classify_initial_support(raw_contact, mjx_floor_contact, raw_foot_z,
                                       allow_airborne=args.allow_airborne_s0)
    initial_support_ok = support["support_ok"]
    feet_penetration_ok = bool(np.isfinite(raw_foot_z).all() and raw_foot_z.min() >= -2e-3)
    upright = float(1.0 - 2.0 * (q[4] * q[4] + q[5] * q[5])) if np.isfinite(q).all() else -1.0
    physically_valid = bool(np.isfinite(q).all() and np.isfinite(qv).all()
                            and abs(qnorm - 1.0) <= 1e-3 and not joint_bad
                            and initial_support_ok and feet_penetration_ok
                            and upright > env.termination_fellover_threshold
                            and raw_nonfoot_pen <= 2e-3 and raw_self_pen <= 2e-3
                            and abs(z_shift) <= 1e-9)
    initial = dict(raw_four_foot_contact=bool(raw_contact.all()), raw_contact=raw_contact.tolist(),
                   raw_foot_contact_count=int(raw_contact.sum()),
                   raw_foot_z_m=raw_foot_z.tolist(),
                   raw_feet_penetration_ok=feet_penetration_ok,
                   raw_nonfoot_penetration_m=raw_nonfoot_pen,
                   raw_self_penetration_m=raw_self_pen, raw_joint_limit_violation=bool(joint_bad),
                   raw_quaternion_norm=qnorm, raw_upright_cosine=upright,
                   initial_support_ok=bool(initial_support_ok),
                   initial_linear_speed_m_s=float(np.linalg.norm(qv[:3])),
                   mjx_four_foot_floor_contact=bool(all(mjx_floor_contact)),
                   mjx_foot_floor_contact=mjx_floor_contact,
                   mjx_foot_contact_count=int(sum(mjx_floor_contact)),
                   physically_valid=physically_valid,
                   initial_support_mode=support["initial_support_mode"],
                   preprojection=preprojection, geometry_projection=projection,
                   geometry_adjusted=bool(projection["applied_lift_m"] > 0),
                   executed_body_height_m=float(q[2]),
                   applied_body_z_shift_m=z_shift + projection["applied_lift_m"])
    # Infeasible generated S0 is REJECTED, not passed into open-loop stepping.
    # A rejected run carries an empty validity mask, never a fake zero reward.
    if not physically_valid:
        initial["execution_skipped"] = True
        initial["skip_reason"] = ("geometry_correction_exceeds_limit" if not projection["allowed"]
                                  else "invalid_initial_physics")
        return (np.zeros((EP, X_DIM), np.float32),
                np.zeros(EP, bool), -2, np.asarray([q]), env.mj_model, initial)
    initial["execution_skipped"] = False
    initial["skip_reason"] = None
    off, sc, (lo, hi) = np.asarray(env.action_offset), float(env.action_scale), env.clip_actions
    qs, vs, cs = [], [], []
    fell = -1
    obstacle_contact = False
    for t in range(EP):
        qs.append(np.asarray(data.qpos)), vs.append(np.asarray(data.qvel))
        geom, dist = np.asarray(data._impl.contact.geom), np.asarray(data._impl.contact.dist)
        if args.task == "obstacle":
            obstacle_contact |= pole_contact(env.mj_model, geom, dist)
        # S[38:42] contact means FOOT-FLOOR, not foot-any-geometry.
        cs.append([((((geom[:, 0] == f) & (geom[:, 1] == floor)) |
                      ((geom[:, 1] == f) & (geom[:, 0] == floor))) &
                     (dist <= 1e-3)).any() for f in feet])
        w, x, y, _ = qs[-1][3:7]
        if 1 - 2 * (x * x + y * y) < env.termination_fellover_threshold:  # 몸통 z 축의 world z 성분
            fell = t
            break
        if t == EP - 1:
            break
        if replan is not None and t > 0 and t % args.replan == 0:
            new = replan(t, qs[-1], vs[-1], cs[-1])
            tq = tq.copy()
            tq[t:t + len(new)] = new
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
    if args.task == "obstacle":
        goal_distance = float(np.linalg.norm(x[L-1, :2] - reward.goal))
        initial["obstacle_task"] = dict(contact=obstacle_contact, goal_distance_m=goal_distance,
                                        success=bool(not obstacle_contact and fell == -1 and goal_distance <= .2))
    return x, m, fell, np.array(qs), env.mj_model, initial


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


summary = dict(s_gradient_scale=0.0 if args.no_guide_s else args.s_gradient_scale, seed=args.seed,
               reward=reward.describe(), task=args.task, ckpt=str(args.ckpt), s0_motion=None if GEN_S0 else args.s0_motion,
               candidates=args.candidates, seconds=args.seconds, replan=args.replan, no_s0_align=args.no_s0_align,
               geometry_s0=args.geometry_s0, max_geometry_lift_m=args.max_geometry_lift_m,
               allow_airborne_s0=args.allow_airborne_s0,
               shared_e_guidance=args.shared_e_guidance, joint_full_guidance=args.joint_full_guidance,
               initial_condition=(f"E-derived expert walk {args.expert_warmup_s0:g}s (S0 = g(E))" if args.expert_warmup_s0 > 0 else
                                  "E-derived home stance (S0 = g(E))" if args.e_rest_s0 else
                                  "xy-yaw-anchor-v4" if args.rest_s0 else "free" if args.free_s0 else "observed"),
               shared_e_method=("projected SPSA on full TT-predicted horizon reward" if args.shared_e_guidance else
                                "per-step full-horizon gradient on shared noisy E and S/A" if args.joint_full_guidance
                                else "first-window E only"),
               shared_e_settings=(dict(iters=args.shared_e_iters, perturb=args.shared_e_perturb,
                                       step_size=args.shared_e_step_size, noise_seed=args.seed + 731)
                                  if args.shared_e_guidance else None), runs={})
fig, axes = plt.subplots(1, len(args.scales), figsize=(5 * len(args.scales), 5), squeeze=False)
for si, scale in enumerate(args.scales):
    optimizer_history = None
    E_initial = None
    joint_audit = None
    if args.fixed_e:
        traj, E = generate_fixed_designs(scale)
    elif args.joint_full_guidance:
        # E is not chosen first or optimized after generation. At *each*
        # reverse diffusion step, the full-episode differentiable reward
        # guides the still-noisy morphology token and all S/A tokens.
        traj, E, joint_audit = joint_diffusion_full_reward(
            net, reward, n=args.candidates, episode_states=EP,
            horizon=H, state_dim=S_DIM, action_dim=X_DIM - S_DIM, e_dim=E_DIM,
            mu=mu, sd=sd, beta=beta, abar=abar, emu=emu, esd=esd,
            guidance_scale=scale, clip=CLIP, seed=args.seed,
            no_guide_s=args.no_guide_s, s_gradient_scale=args.s_gradient_scale,
            s0_fn=(warm_s0_of if args.expert_warmup_s0 > 0 else rest_s0_of) if args.e_rest_s0 else None)
    elif args.shared_e_guidance:
        # Scale 0 is the unoptimized baseline. All other scales start from
        # exactly the same prior morphology E, and compare full 10s rewards.
        E_initial = sample_rest_morphology(args.candidates, args.seed)
        eval_seed = args.seed + 731
        def full_horizon_scores(e):
            sample = generate_fixed_e(e, scale, eval_seed)
            mask = torch.ones(len(e), EP, dtype=torch.bool, device=dev)
            return reward(sample, mask)[0]
        E, best_score, optimizer_history = optimize_shared_e(
            E_initial, full_horizon_scores,
            steps=(args.shared_e_iters if scale > 0 else 0),
            perturb=args.shared_e_perturb, step_size=args.shared_e_step_size,
            seed=args.seed + 2718)
        traj = generate_fixed_e(E, scale, eval_seed)
        print(f"shared E scale={scale:g}: full-horizon predicted rewards {best_score.cpu().tolist()}", flush=True)
    else:
        traj, E = generate(scale, args.candidates, args.seed)
    pred = traj.cpu().numpy()
    pm = np.ones((args.candidates, EP), bool)
    p_total, p_terms = score(pred, pm)
    rows = []
    mj = np.zeros_like(pred)
    mm = np.zeros_like(pm)
    for i in range(args.candidates):
        def replan(t, q, v, c, i=i):
            # 실제 MJX state 에서 E 고정 window 재생성. 실행되는 계획을 pred 에 기록 (예측 = 실행한 계획).
            cur = torch.zeros(1, X_DIM, device=dev)
            cur[0, :S_DIM] = torch.as_tensor(features(q[None], v[None], np.asarray(c)[None])[0], device=dev)
            e_norm = ((E[i:i + 1].to(dev) - emu) / esd).float()
            ep = plan_window(cur, e_norm, scale, args.seed + 7919 * i + t, min(H, EP - t))[0].cpu().numpy()
            pred[i, t:t + len(ep)] = ep
            return ep[:, A]
        mj[i], mm[i], fell, qraw, model, initial = execute(E[i].cpu().numpy(), pred[i, :, A], pred[i, 0, :S_DIM],
                                                          replan=replan if args.replan else None)
        name = f"scale_{scale:g}_cand{i}"
        save_design(model, E[i].cpu().numpy(), name)
        if not args.no_render and not initial.get("execution_skipped"):
            render(model, qraw, pred[i], args.out / "videos" / f"{name}.mp4")
        rows.append((fell, initial))
    if args.replan:
        p_total, p_terms = score(pred, pm)  # pred = 실행한 재계획 window 들
    r_total, r_terms = score(mj, mm)
    L = mm.sum(1)
    path_err = [float(np.linalg.norm(pred[i, :L[i], :2] - mj[i, :L[i], :2], axis=-1).mean())
                if L[i] > 0 else None for i in range(args.candidates)]
    res = []
    for i in range(args.candidates):
        invalid = bool(rows[i][1].get("execution_skipped", False))
        res.append(dict(E=E[i].cpu().numpy().round(3).tolist(),
                        pred_total=float(p_total[i]), pred_progress=float(p_terms["r_progress"][i] * reward.unit_scale),
                        mjx_total=None if invalid else float(r_total[i]),
                        mjx_progress=None if invalid else float(r_terms["r_progress"][i] * reward.unit_scale),
                        mjx_fell_step=None if invalid else int(rows[i][0]),
                        mjx_valid_s=float(L[i] * 0.02), path_err_m=path_err[i],
                        initial_physics=rows[i][1],
                        physically_valid_s0=bool(rows[i][1]["physically_valid"]),
                        execution_skipped=invalid,
                        skip_reason=rows[i][1].get("skip_reason"),
                        pred_terms={k: float(v[i]) for k, v in p_terms.items()},
                        mjx_terms=None if invalid else {k: float(v[i]) for k, v in r_terms.items()}))
        if invalid:
            print(f"scale {scale:g} #{i}: MJX 실행 제외 - {res[-1]['skip_reason']} | "
                  f"TT 예상 Reward {p_total[i]:.2f} (실제 Reward는 평가하지 않음)", flush=True)
        else:
            print(f"scale {scale:g} #{i}: TT 예측 {res[-1]['pred_progress']:.2f} {reward.unit} (R {p_total[i]:.2f}) | "
                  f"MJX {res[-1]['mjx_progress']:.2f} {reward.unit} (R {r_total[i]:.2f}), "
                  f"낙상 {rows[i][0]}, 경로 차 {path_err[i]:.2f} m | E[:3] {res[-1]['E'][:3]}", flush=True)
    summary["runs"][f"scale_{scale:g}"] = res
    if joint_audit is not None:
        summary.setdefault("joint_full_guidance_audit", {})[f"scale_{scale:g}"] = joint_audit
    # Controlled same-scale comparison: changes ONLY the shared E search,
    # not reward weights, local DDIM guidance, noise seed, or candidate prior.
    if args.shared_e_guidance and args.shared_e_eval_baseline and scale > 0:
        baseline_traj = generate_fixed_e(E_initial, scale, args.seed + 731)
        base_pred = baseline_traj.cpu().numpy()
        base_p, _ = score(base_pred, np.ones((args.candidates, EP), bool))
        base_mjx = np.zeros_like(base_pred)
        base_mask = np.zeros((args.candidates, EP), bool)
        baseline_rows = []
        for i in range(args.candidates):
            base_mjx[i], base_mask[i], base_fell, _, _, base_initial = execute(
                E_initial[i].cpu().numpy(), base_pred[i, :, A], base_pred[i, 0, :S_DIM])
            baseline_rows.append(dict(pred_total=float(base_p[i]), fell_step=int(base_fell),
                                      physically_valid_s0=bool(base_initial["physically_valid"])))
        base_r, _ = score(base_mjx, base_mask)
        for i in range(args.candidates):
            baseline_rows[i]["mjx_total"] = float(base_r[i])
        summary.setdefault("shared_e_control", {})[f"scale_{scale:g}"] = baseline_rows
        np.savez_compressed(args.out / f"scale_{scale:g}_unoptimized_same_scale.npz",
                            pred=base_pred, mjx=base_mjx, mjx_mask=base_mask,
                            E=E_initial.cpu().numpy())
    if optimizer_history is not None:
        summary.setdefault("shared_e_history", {})[f"scale_{scale:g}"] = optimizer_history
        summary.setdefault("shared_e_initial", {})[f"scale_{scale:g}"] = E_initial.cpu().numpy().tolist()
    np.savez_compressed(args.out / f"scale_{scale:g}.npz", pred=pred, mjx=mj, mjx_mask=mm, E=E.cpu().numpy())
    ax = axes[0, si]
    for pts, center in reward.guide_lines():
        ax.plot(pts[:, 0], pts[:, 1], "k-" if center else "k:", lw=1 if center else 0.7)
    for i in range(args.candidates):
        c = plt.cm.tab10(i % 10)
        ax.plot(pred[i, :, 0], pred[i, :, 1], "--", color=c, lw=1)
        ax.plot(mj[i, :L[i], 0], mj[i, :L[i], 1], "-", color=c, lw=1.5)
        if rows[i][0] >= 0:
            ax.plot(mj[i, L[i] - 1, 0], mj[i, L[i] - 1, 1], "x", color=c, ms=8)
    ax.set_aspect("equal")
    ax.set_title(f"guidance {scale:g} (dashed: TT, solid: MJX, x: fall)")
json.dump(summary, open(args.out / "summary.json", "w"), indent=1, ensure_ascii=False)
fig.tight_layout()
fig.savefig(args.out / "paths.png", dpi=110)
print("->", args.out)
