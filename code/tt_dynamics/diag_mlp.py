"""진단: diffusion 없이 1-step MLP (E, S_t, A_t) -> S_{t+1} 로 같은 rollout 의 dynamics 를 얼마나 맞힐 수 있는지.

TT (train_tt.py) 와 같은 특징·같은 train/val 분할을 쓰고, 같은 지표로 비교한다.
  - 단일 window: 검증 episode 시작점 8 개, 실제 s0 에서 실제 action 으로 31 step open-loop (TT --diag 와 같은 지표)
  - 전체 episode: 검증 episode 앞 64 개, 500 step open-loop (TT eval_dynamics 와 같은 지표)
usage (codesign/ 에서): CUDA_VISIBLE_DEVICES=0 ../transformer-transformer/.venv/bin/python code/tt_dynamics/diag_mlp.py --out runs/pipeline_v1/tt/mlp/try1
"""

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rollout_data import S_DIM, load

INV = [2] + list(range(5, S_DIM))  # 위치·yaw 와 무관한 state (높이, 중력, 속도, 각속도, 관절, 접지)
Q, VEL, WZ = slice(14, 26), slice(8, 10), 13

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", nargs="+", default=sorted(glob.glob("runs/pipeline_v1/rollouts/e00[0-2]/*_seed0.npz")))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--steps", type=int, default=20000)
parser.add_argument("--batch", type=int, default=4096)
parser.add_argument("--hidden", type=int, default=512)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--val-frac", type=float, default=0.1)
args = parser.parse_args()
dev = "cuda"
torch.manual_seed(args.seed)
EP, H = 500, 32

X, E, M, src, CL = load(args.data, EP)
perm = np.random.default_rng(args.seed).permutation(len(X))  # train_tt.py 와 같은 분할
n_val = max(1, int(len(X) * args.val_frac))
val, tr = perm[:n_val], perm[n_val:]


def yaw(x):
    return np.arctan2(x[..., 4], x[..., 3])


def step_motion(x):
    """episode 기준 [N,T,X] -> t 에서 t+1 로의 (dx, dy: t 의 heading frame, dyaw) [N,T-1,3]."""
    y = yaw(x)
    dp = x[:, 1:, 0:2] - x[:, :-1, 0:2]
    c, s = np.cos(y[:, :-1]), np.sin(y[:, :-1])
    return np.stack([c * dp[..., 0] + s * dp[..., 1], -s * dp[..., 0] + c * dp[..., 1],
                     np.angle(np.exp(1j * (y[:, 1:] - y[:, :-1])))], -1)


def pairs(idx):
    x, m = X[idx], M[idx]
    ok = m[:, :-1] & m[:, 1:]
    inp = np.concatenate([np.repeat(E[idx][:, None], EP - 1, 1), x[:, :-1, INV], x[:, :-1, S_DIM:]], -1)[ok]
    out = np.concatenate([x[:, 1:, INV] - x[:, :-1, INV], step_motion(x)], -1)[ok]
    return inp.astype(np.float32), out.astype(np.float32)


xi, yo = pairs(tr)
xv, yv = pairs(val)
imu, isd = xi.mean(0), xi.std(0) + 1e-6
omu, osd = yo.mean(0), yo.std(0) + 1e-6
norm = [torch.tensor(a, device=dev) for a in (imu, isd, omu, osd)]
Xi, Yo = torch.tensor((xi - imu) / isd, device=dev), torch.tensor((yo - omu) / osd, device=dev)
Xv, Yv = torch.tensor((xv - imu) / isd, device=dev), torch.tensor((yv - omu) / osd, device=dev)
net = nn.Sequential(nn.Linear(xi.shape[1], args.hidden), nn.SiLU(), nn.Linear(args.hidden, args.hidden), nn.SiLU(),
                    nn.Linear(args.hidden, args.hidden), nn.SiLU(), nn.Linear(args.hidden, yo.shape[1])).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
args.out.mkdir(parents=True, exist_ok=True)
log = open(args.out / "progress.jsonl", "w")
t0 = time.time()
print(f"transitions train {len(xi)} val {len(xv)}")
for step in range(1, args.steps + 1):
    b = torch.randint(0, len(Xi), (args.batch,), device=dev)
    loss = ((net(Xi[b]) - Yo[b]) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 1000 == 0:
        with torch.no_grad():
            vl = ((net(Xv) - Yv) ** 2).mean().item()
        rec = dict(step=step, loss=loss.item(), val_loss=vl, min=(time.time() - t0) / 60)
        log.write(json.dumps(rec) + "\n")
        print(rec, flush=True)
net.eval()


@torch.no_grad()
def rollout(s0, e, acts):
    """s0 [N,X] (위치·yaw 포함), e [N,13], acts [N,T,12] -> 예측 state [N,T,S_DIM] (s0 기준 위치·yaw)."""
    s = torch.tensor(s0[:, :S_DIM], device=dev)
    e, acts = torch.tensor(e, device=dev), torch.tensor(acts, device=dev)
    p, y = s[:, 0:2].clone(), torch.atan2(s[:, 4], s[:, 3])
    out = [s]
    for t in range(acts.shape[1] - 1):
        inp = torch.cat([e, s[:, INV], acts[:, t]], -1)
        d = net((inp - norm[0]) / norm[1]) * norm[3] + norm[2]
        c, sn = torch.cos(y), torch.sin(y)
        p = p + torch.stack([c * d[:, -3] - sn * d[:, -2], sn * d[:, -3] + c * d[:, -2]], -1)
        y = y + d[:, -1]
        s = s.clone()
        s[:, INV] = s[:, INV] + d[:, :len(INV)]
        s[:, 0:2], s[:, 3], s[:, 4] = p, torch.cos(y), torch.sin(y)
        out.append(s)
    return torch.stack(out, 1).cpu().numpy()


wrap = lambda a: np.angle(np.exp(1j * a))  # noqa: E731
res = {}
# 1-step
with torch.no_grad():
    d = (net(Xv) * norm[3] + norm[2]).cpu().numpy()
qi = [INV.index(i) for i in range(14, 26)]
vi = [INV.index(i) for i in (8, 9)]
res["one_step"] = dict(
    joint_err_deg=float(np.degrees(np.sqrt(((d[:, qi] - yv[:, qi]) ** 2).mean()))),
    joint_hold_deg=float(np.degrees(np.sqrt((yv[:, qi] ** 2).mean()))),
    vel_err=float(np.linalg.norm(d[:, vi] - yv[:, vi], axis=-1).mean()), vel_hold=float(np.linalg.norm(yv[:, vi], axis=-1).mean()),
    pos_err_cm=float(np.linalg.norm(d[:, -3:-1] - yv[:, -3:-1], axis=-1).mean() * 100),
    pos_hold_cm=float(np.linalg.norm(yv[:, -3:-1], axis=-1).mean() * 100),
    dyaw_err_deg=float(np.degrees(np.abs(d[:, -1] - yv[:, -1]).mean())), dyaw_hold_deg=float(np.degrees(np.abs(yv[:, -1]).mean())))

# 단일 window (TT --diag 와 같은 지표)
rng = np.random.default_rng(args.seed)
vi_ep = np.repeat(val, 8)
L = M[vi_ep].sum(1)
vi_ep, L = vi_ep[L >= H], L[L >= H]
st = (rng.random(len(vi_ep)) * (L - H + 1)).astype(int)
tt = st[:, None] + np.arange(H)
win = X[vi_ep[:, None], tt].copy()
# window 첫 state 기준으로 위치·yaw 변환
y0 = yaw(win[:, :1])
c0, s0 = np.cos(y0), np.sin(y0)
dp = win[..., 0:2] - win[:, :1, 0:2]
yw = yaw(win) - y0
win[..., 0], win[..., 1] = c0 * dp[..., 0] + s0 * dp[..., 1], -s0 * dp[..., 0] + c0 * dp[..., 1]
win[..., 3], win[..., 4] = np.cos(yw), np.sin(yw)
P = rollout(win[:, 0], E[vi_ep], win[..., S_DIM:])
T = win[..., :S_DIM]
lab = CL[vi_ep, st]
kind = np.array(["정지" if ("squat" in c or "tilt" in c) else "회전" if ("CCW" in c or "CW" in c) else "직진·횡이동" for c in lab])
k = H - 1
for name in ("정지", "직진·횡이동", "회전", "전체"):
    g = kind == name if name != "전체" else np.ones(len(kind), bool)
    p, t_ = P[g], T[g]
    dy_p, dy_t = np.degrees(wrap(yaw(p[:, k]))), np.degrees(wrap(yaw(t_[:, k])))
    res[name] = dict(
        n=int(g.sum()), yaw_err_deg=float(np.degrees(np.abs(wrap(np.radians(dy_p - dy_t))).mean())), yaw_hold_deg=float(np.abs(dy_t).mean()),
        yaw_corr=float(np.corrcoef(dy_t, dy_p)[0, 1]), yaw_slope=float(np.polyfit(dy_t, dy_p, 1)[0]),
        vel_err=float(np.linalg.norm(p[:, 1:, VEL] - t_[:, 1:, VEL], axis=-1).mean()),
        vel_hold=float(np.linalg.norm(t_[:, :1, VEL] - t_[:, 1:, VEL], axis=-1).mean()),
        wz_err=float(np.abs(p[:, 1:, WZ] - t_[:, 1:, WZ]).mean()), wz_hold=float(np.abs(t_[:, :1, WZ] - t_[:, 1:, WZ]).mean()),
        pos_err_cm=float(np.linalg.norm(p[:, k, :2] - t_[:, k, :2], axis=-1).mean() * 100),
        pos_hold_cm=float(np.linalg.norm(t_[:, k, :2], axis=-1).mean() * 100),
        joint_err_deg=float(np.degrees(np.sqrt(((p[:, 1:, Q] - t_[:, 1:, Q]) ** 2).mean()))))

# 전체 episode (TT eval_dynamics 와 같은 64 개)
ev = val[:64]
Pe = rollout(X[ev, 0], E[ev], X[ev, :, S_DIM:])
Te, me = X[ev, :, :S_DIM], M[ev]
for t in (1, 10, 50, 100, 250):
    ok = me[:, t]
    res[f"episode_t{t}"] = dict(n=int(ok.sum()),
                                joint_rmse_deg=float(np.degrees(np.sqrt(((Pe[ok, t, Q] - Te[ok, t, Q]) ** 2).mean()))),
                                pos_err_cm=float(np.linalg.norm(Pe[ok, t, :2] - Te[ok, t, :2], axis=-1).mean() * 100),
                                yaw_err_deg=float(np.degrees(np.abs(wrap(yaw(Pe[ok, t]) - yaw(Te[ok, t]))).mean())))
res["episode_all"] = dict(joint_rmse_deg=float(np.degrees(np.sqrt(((Pe[..., Q] - Te[..., Q]) ** 2)[me].mean()))),
                          pos_err_cm=float(np.linalg.norm(Pe[..., :2] - Te[..., :2], axis=-1)[me].mean() * 100))
json.dump(res, open(args.out / "diag_mlp.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))
