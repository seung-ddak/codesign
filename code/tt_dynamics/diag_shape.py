"""D4 형상 민감도: 같은 s0·A 에 E 만 e000/e001/e002 로 바꿨을 때 TT 예측의 변화가 MJX 실제 변화와 같은 방향인가.

window (H step) 를 rollout 데이터 (3 형상 seq_seed0) 에서 뽑아, 그 시작 qpos·qvel 과 target_q 를 고정한다.
형상마다: MJX = 그 E 로 만든 모델에서 target_q 를 open-loop 실행 (발이 바닥에 닿게 높이만 맞춤).
          TT  = mode 1 (E·s0·A 고정) 로 S 생성. s0 는 같은 형상의 MJX 첫 state. 형상끼리 같은 초기 노이즈, seed K 개 평균.
지표 (window 끝, window 기준 좌표): 전진 dx, 횡 dy, yaw, 평균 높이, 평균 pitch·roll (중력 x·y).
형상 쌍 (1-0, 2-0, 2-1) 의 차이 Δ 를 TT·MJX 로 구해 window 간 상관을 본다. MJX 가 넘어진 window 는 뺀다.
--vary A (D5): E 는 원 형상으로 고정하고 A 만 바꾼다 (같은 형상 다른 window 의 A / 관절별 상수 offset). 쌍 = swap-orig, offset-orig, offset-swap.

usage (codesign/ 에서):
  CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false ../transformer-transformer/.venv/bin/python \
      code/tt_dynamics/diag_shape.py --ckpt runs/pipeline_v1/tt/h32_try6 --out runs/pipeline_v1/diag/d4_try1
"""

import argparse
import json
import math
import pickle
import sys
from pathlib import Path

CODESIGN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODESIGN.parent / "transformer-transformer"))
sys.path.insert(0, str(CODESIGN / "code" / "motion_tracking"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mujoco import mjx  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pd_replay import build_model  # noqa: E402
from rollout_data import E_DIM, S_DIM, features  # noqa: E402
from t2.mjx_env.base_env import make_data, step  # noqa: E402
from track_env import TrackMjxEnv  # noqa: E402
from tt_model import Net, condition_masks, schedule  # noqa: E402

FOOT_R = 0.02
SHAPES = ["e000", "e001", "e002"]
PAIRS = [(1, 0), (2, 0), (2, 1)]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--windows", type=int, default=64, help="데이터 형상당 window 수")
parser.add_argument("--tt-seeds", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--vary", choices=("E", "A"), default="E", help="A = D5: 원 형상 E 고정, A 만 바꿈 (다른 window 의 A / 상수 offset)")
parser.add_argument("--offset", type=float, default=0.1, help="--vary A 의 관절별 상수 offset 최대 (rad)")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
dev = "cuda"

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


@torch.no_grad()
def tt_dynamics(x, E, seed):
    """x [B,H,X] window 기준 (s0·A 사용), E [13] -> S 예측 [B,H,X] (mode 1, DDIM)."""
    B = len(x)
    x0 = (torch.as_tensor(x, device=dev) - mu) / sd
    e0 = ((torch.as_tensor(E, device=dev) - emu) / esd).expand(B, E_DIM)
    kx, ke = condition_masks(torch.full((B,), 1, device=dev), torch.ones(B, H, dtype=torch.bool, device=dev))
    pad = torch.zeros(B, H, dtype=torch.bool, device=dev)
    xt = torch.randn(x0.shape, generator=torch.Generator(dev).manual_seed(seed), device=dev)
    for k in reversed(range(len(beta))):
        ab, ab_prev = abar[k], (abar[k - 1] if k > 0 else torch.ones((), device=dev))
        xt = torch.where(kx > 0, x0, xt)
        ex, _ = net(xt, e0, kx, ke, torch.full((B,), k, device=dev, dtype=torch.float32), pad)
        x0h = ((xt - (1 - ab).sqrt() * ex) / ab.sqrt()).clamp(-CLIP, CLIP)
        xt = ab_prev.sqrt() * x0h + (1 - ab_prev).sqrt() * (xt - ab.sqrt() * x0h) / (1 - ab).sqrt()
    return (torch.where(kx > 0, x0, xt) * sd + mu).cpu().numpy()


# ---- 데이터 window ----
rng = np.random.default_rng(args.seed)
D = {n: np.load(CODESIGN / f"runs/pipeline_v1/rollouts/{n}/seq_seed0.npz") for n in SHAPES}
Es = [D[n]["E"].astype(np.float32) for n in SHAPES]
W = []  # (원 형상, qpos0, qvel0, target_q [H,12])
for si, n in enumerate(SHAPES):
    v = D[n]["valid"]
    ok = [(b, t) for b in range(len(v)) for t in range(0, v.shape[1] - H, 8) if v[b, t:t + H].all()]
    for j in rng.choice(len(ok), args.windows, replace=False):
        b, t = ok[j]
        W.append((si, D[n]["qpos"][b, t], D[n]["qvel"][b, t], D[n]["target_q"][b, t:t + H]))
src = np.array([w[0] for w in W])
TQ = np.stack([w[3] for w in W])
NW = len(W)

# ---- MJX (window 를 vmap 으로 묶어 실행) ----
expert = CODESIGN / "runs/pipeline_v1/experts/walkForwardSlow__r00"
cfg = pickle.load(open(sorted(expert.glob("wandb/offline-run-*/files/config.pkl"))[-1], "rb"))
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
choice = cfg.get("choice")
ecfg["reference_dir"] = str(CODESIGN / "runs/pipeline_v1/retargeting/e000" / choice)
ecfg["clips"] = ["walkForwardSlow__r00"]


def mjx_run(E, W, TQ):
    """window 들을 형상 E 로 실행 -> 특징 [N,H,X] (window 기준), 낙상 여부 [N]."""
    NW = len(W)
    env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                                     torque_limit=cfg.get("torque_limit"), uniforms=E), **ecfg)
    feet = np.asarray(env._foot_ids)
    d = mujoco.MjData(env.mj_model)
    datas = []
    for _, q, qv, _ in W:
        q = q.copy()
        d.qpos[:] = q
        mujoco.mj_kinematics(env.mj_model, d)
        q[2] -= d.geom_xpos[feet, 2].min() - FOOT_R
        datas.append(mjx.forward(env.mjx_model, make_data(env.mj_model, qpos=jnp.asarray(q), qvel=jnp.asarray(qv), impl=env.impl,
                                                           nconmax=env._nconmax, njmax=env._njmax)))
    data = jax.tree.map(lambda *a: jnp.stack(a), *datas)
    vstep = jax.jit(jax.vmap(lambda dd, c: step(env.mjx_model, dd, c, env.n_substeps)))
    off, sc, (lo, hi) = np.asarray(env.action_offset), float(env.action_scale), env.clip_actions
    qs, vs, cs = [], [], []
    for t in range(H):
        qs.append(np.asarray(data.qpos)), vs.append(np.asarray(data.qvel))
        geom, dist = np.asarray(data._impl.contact.geom), np.asarray(data._impl.contact.dist)
        cs.append(np.stack([((geom == f).any(-1) & (dist < 1e-3)).any(-1) for f in feet], -1))
        if t == H - 1:
            break
        ctrl = np.tile(off, (NW, 1))
        ctrl[:, :12] = off[:12] + np.clip((TQ[:, t] - off[:12]) / sc, lo, hi) * sc
        data = vstep(data, jnp.asarray(ctrl))
    qs, vs, cs = np.stack(qs, 1), np.stack(vs, 1), np.stack(cs, 1)
    x = np.concatenate([np.stack([features(qs[i], vs[i], cs[i]) for i in range(NW)]), TQ], -1)
    up = 1 - 2 * (qs[..., 4] ** 2 + qs[..., 5] ** 2)  # 몸통 z 축의 world z 성분
    return x, (up < env.termination_fellover_threshold).any(1)


def metrics(x):
    """[N,H,X] -> 지표 dict, 각 [N]."""
    e = x[:, -1]
    return dict(dx=e[:, 0], dy=e[:, 1], yaw_deg=np.degrees(np.arctan2(e[:, 4], e[:, 3])), z=x[:, :, 2].mean(1),
                gx=x[:, :, 5].mean(1), gy=x[:, :, 6].mean(1))


M_mjx, M_tt, fell = [], [], []
if args.vary == "E":
    VARIANTS = SHAPES
    for si, E in enumerate(Es):
        xm, f = mjx_run(E, W, TQ)
        xt = np.mean([tt_dynamics(xm, E, args.seed + k) for k in range(args.tt_seeds)], 0)  # s0·A 는 MJX 와 같은 값
        M_mjx.append(metrics(xm)), M_tt.append(metrics(xt)), fell.append(f)
        print(SHAPES[si], "MJX 낙상", int(f.sum()), "/", NW, flush=True)
else:  # D5: 같은 s0·E (원 형상), A = 원래 / 같은 형상 다른 window 의 A / 원래 + 관절별 상수 offset
    VARIANTS = ["orig", "swap", "offset"]
    perm = np.concatenate([rng.permutation(np.where(src == i)[0]) for i in range(3)])
    swap = np.empty(NW, int)
    swap[np.concatenate([np.where(src == i)[0] for i in range(3)])] = perm
    TQs = [TQ, TQ[swap], TQ + rng.uniform(-args.offset, args.offset, (NW, 1, TQ.shape[-1])).astype(np.float32)]
    for vi, tq in enumerate(TQs):
        xm, xt, f = np.zeros((NW, H, TQ.shape[-1] + S_DIM), np.float32), None, np.zeros(NW, bool)
        xt = np.zeros_like(xm)
        for si, E in enumerate(Es):
            idx = np.where(src == si)[0]
            xm[idx], f[idx] = mjx_run(E, [W[i] for i in idx], tq[idx])
            xt[idx] = np.mean([tt_dynamics(xm[idx], E, args.seed + k) for k in range(args.tt_seeds)], 0)
        M_mjx.append(metrics(xm)), M_tt.append(metrics(xt)), fell.append(f)
        print(VARIANTS[vi], "MJX 낙상", int(f.sum()), "/", NW, flush=True)
SHAPES_OUT = VARIANTS

keep = ~np.any(fell, 0)
res = dict(ckpt=str(args.ckpt), windows=NW, kept=int(keep.sum()), tt_seeds=args.tt_seeds,
           fell={SHAPES_OUT[i]: int(fell[i].sum()) for i in range(3)}, pairs={}, abs_err={})
for a, b in PAIRS:
    key = f"{SHAPES_OUT[a]}-{SHAPES_OUT[b]}"
    res["pairs"][key] = {}
    for m in M_mjx[0]:
        dm, dt = (M_mjx[a][m] - M_mjx[b][m])[keep], (M_tt[a][m] - M_tt[b][m])[keep]
        res["pairs"][key][m] = dict(corr=float(np.corrcoef(dm, dt)[0, 1]), sign=float((np.sign(dm) == np.sign(dt)).mean()),
                                    slope=float((dm * dt).sum() / (dm * dm).sum()), mjx_mean=float(dm.mean()), tt_mean=float(dt.mean()),
                                    mjx_std=float(dm.std()))
# TT 절대 오차: 데이터 원 형상과 같은 E (학습 분포) vs 다른 E (반사실)
for m in M_mjx[0]:
    err = [np.abs(M_tt[i][m] - M_mjx[i][m])[keep] for i in range(3)]
    if args.vary == "E":  # own = 데이터 원 형상 E (학습 분포), cross = 다른 형상 E (반사실)
        res["abs_err"][m] = dict(own=float(np.concatenate([err[i][src[keep] == i] for i in range(3)]).mean()),
                                 cross=float(np.concatenate([err[i][src[keep] != i] for i in range(3)]).mean()))
    else:  # own = 원래 A, cross = 바꾼 A (swap·offset)
        res["abs_err"][m] = dict(own=float(err[0].mean()), cross=float(np.concatenate(err[1:]).mean()))
json.dump(res, open(args.out / f"d{4 if args.vary == 'E' else 5}.json", "w"), indent=1, ensure_ascii=False)
np.savez_compressed(args.out / f"d{4 if args.vary == 'E' else 5}.npz", src=src, keep=keep, fell=np.stack(fell),
                    **{f"{w}_{SHAPES_OUT[i]}_{m}": M[i][m] for w, M in (("mjx", M_mjx), ("tt", M_tt)) for i in range(3) for m in M[i]})
for k, v in res["pairs"].items():
    print(k, {m: f"r {d['corr']:+.2f} 부호 {d['sign']:.2f} 기울기 {d['slope']:+.2f} (MJX Δ {d['mjx_mean']:+.3f}±{d['mjx_std']:.3f}, TT Δ {d['tt_mean']:+.3f})"
              for m, d in v.items()})
print("abs_err", {m: f"{d['own']:.3f}/{d['cross']:.3f}" for m, d in res["abs_err"].items()})
print("->", args.out)
