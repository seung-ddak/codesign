"""(repro_check.py 에서 가져온 공통부) H64 TT 재현 확인: 넘어지지 않은 / 넘어진 검증 에피소드.

A. 동역학 예측 (Mode 1): 실제 E, s0, A 를 주고 TT 가 10 초 상태를 예측 -> 실제와 비교.
B. 생성 후 실행 (Mode 2): 실제 E, s0 만 주고 TT 가 S, A 를 생성 -> A 를 MuJoCo 에서 실행.
C. 확인용: 실제 A 를 같은 MuJoCo 재생기로 실행 -> 수집 때의 생존 여부와 일치하는지.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

CODE = Path("/home/cgr-ugrad-2026/quad_ws/codesign/code/tt_dynamics")
sys.path.insert(0, str(CODE))
from rollout_data import features, S_DIM, X_DIM  # noqa: E402
from tt_model import make_net, schedule, condition_masks, rebase, unbase  # noqa: E402
import inspect, generate_task_mujoco as gtm  # noqa: E402

# 수집 데이터는 발 1 개 접지 자세로 시작하는 경우가 많다. task 평가기의 "발 2 개 이상 접지" 시작 조건만 뺀다.
# 넘어짐 판정·물리·제어는 그대로다.
_src = inspect.getsource(gtm.replay)
_old = "np.min(raw_clearance) < -.002 or fallen() or support.sum() < 2"
assert _old in _src
exec(_src.replace(_old, "fallen()"), gtm.__dict__)
replay = gtm.replay


# ---- e0000 TT 의 예측 지평별 정확도 (Mode 1: 실제 행동을 주고 상태 예측) ----
p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, default=Path(__file__).resolve().parents[1] / "tt/checkpoint.pt")
p.add_argument("--data", type=Path, default=Path("/home/cgr-ugrad-2026/quad_ws/codesign/runs/calm_dynamics_1000x100/rollouts/e0000/primitive_seed0.npz"))
p.add_argument("--samples", type=int, default=3)
args = p.parse_args()
torch.set_num_threads(8)
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False); cfg = ck["args"]
H, clip = cfg["horizon"], cfg["clip"]
net = make_net(cfg); net.load_state_dict(ck["ema_net"] if ck.get("ema_net") is not None else ck["net"]); net.eval().requires_grad_(False)
mu, sd = torch.as_tensor(ck["mu"]).float(), torch.as_tensor(ck["sd"]).float()
beta, abar = schedule(cfg["diff_steps"], "cpu")
emu, esd = (0.5, 0.5) if cfg["norm"] == "minmax" else (0.5, 0.25)
DELTA = cfg.get("pose", "abs") == "delta"; ar = torch.arange(H)
d = dict(np.load(args.data, allow_pickle=True)); val = np.asarray(ck["val"])
L = 250; STARTS = (0, 100, 200, 300); KS = (25, 50, 100, 249)
seqs, masks, meta = [], [], []
for r in val:
    n = int(d["valid"][r].sum())
    for t0 in STARTS:
        if n - t0 < 26: continue
        k = min(L, n - t0)
        x = np.zeros((L, X_DIM), np.float32); m = np.zeros(L, bool)
        x[:k, :S_DIM] = features(d["qpos"][r, t0:t0 + k], d["qvel"][r, t0:t0 + k], d["contact"][r, t0:t0 + k])
        x[:k, S_DIM:] = d["target_q"][r, t0:t0 + k]; m[:k] = True
        seqs.append(x); masks.append(m); meta.append((int(r), t0, bool(d["fell"][r])))
X = torch.tensor(np.stack(seqs)); M = torch.tensor(np.stack(masks))
En = (torch.tensor(d["E"]).float()[None].expand(len(X), -1) - emu) / esd
def deltas(x):  # train_tt.py 와 같다
    yaw = torch.atan2(x[..., 4], x[..., 3])
    dp = x[:, 1:, 0:2] - x[:, :-1, 0:2]
    c, s = torch.cos(yaw[:, :-1]), torch.sin(yaw[:, :-1])
    dy = torch.remainder(yaw[:, 1:] - yaw[:, :-1] + math.pi, 2 * math.pi) - math.pi
    d = torch.stack([c * dp[..., 0] + s * dp[..., 1], -s * dp[..., 0] + c * dp[..., 1], dy], -1)
    return torch.cat([torch.zeros_like(d[:, :1]), d], 1)


def set_delta(x, d):
    x = x.clone()
    x[..., 0], x[..., 1], x[..., 3], x[..., 4] = d[..., 0], d[..., 1], d[..., 2], 0.0
    return x


def integrate(w, start):
    out = w.clone()
    p, yaw = start[:, 0:2].clone(), torch.atan2(start[:, 4], start[:, 3])
    out[:, 0, 0:2], out[:, 0, 3], out[:, 0, 4] = p, torch.cos(yaw), torch.sin(yaw)
    for i in range(1, w.shape[1]):
        c, s = torch.cos(yaw), torch.sin(yaw)
        p = p + torch.stack([c * w[:, i, 0] - s * w[:, i, 1], s * w[:, i, 0] + c * w[:, i, 1]], -1)
        yaw = yaw + w[:, i, 3]
        out[:, i, 0:2], out[:, i, 3], out[:, i, 4] = p, torch.cos(yaw), torch.sin(yaw)
    return out


def ddim(x0, e0, kx, ke, pad):
    x, e = torch.randn_like(x0), torch.randn_like(e0)
    for k in reversed(range(len(beta))):
        x, e = torch.where(kx > 0, x0, x), torch.where(ke > 0, e0, e)
        kk = torch.full((len(x),), k, dtype=torch.float32)
        ex, ee = net(torch.where(kx > 0, x0, x), torch.where(ke > 0, e0, e), kx, ke, kk, pad)
        ab, abp = abar[k], (abar[k - 1] if k > 0 else torch.ones(()))
        x0h = ((x - (1 - ab).sqrt() * ex) / ab.sqrt()).clamp(-clip, clip)
        e0h = ((e - (1 - ab).sqrt() * ee) / ab.sqrt()).clamp(-clip, clip)
        epx, epe = (x - ab.sqrt() * x0h) / (1 - ab).sqrt(), (e - ab.sqrt() * e0h) / (1 - ab).sqrt()
        x, e = abp.sqrt() * x0h + (1 - abp).sqrt() * epx, abp.sqrt() * e0h + (1 - abp).sqrt() * epe
    return torch.where(kx > 0, x0, x)



DX = deltas(X) if DELTA else None

def rollout():
    n = len(X); pred = X.clone(); pred[:, 1:, :S_DIM] = 0
    cur = X[:, 0].clone(); cur_d = DX[:, 0].clone() if DELTA else None; t0 = 0
    while t0 < L - 1:
        t = (t0 + ar).clamp(max=L - 1); valid = (t0 + ar) < L
        m = valid[None] & M[:, t]
        win = X[:, t].clone(); win[:, 0, :S_DIM] = cur[:, :S_DIM]
        xw = rebase(win)
        if DELTA:
            xw = set_delta(xw, DX[:, t]); xw[:, 0, [0, 1, 3]] = cur_d
        xw = (xw - mu) / sd * m[..., None]
        kx, ke = condition_masks(torch.full((n,), 1, dtype=torch.long), m)
        pw = ddim(xw, En, kx, ke, ~m) * sd + mu
        ep = integrate(pw, cur) if DELTA else unbase(pw, cur)
        k = min(H, L - t0)
        pred[:, t0 + 1:t0 + k, :S_DIM] = ep[:, 1:k, :S_DIM]
        cur = ep[:, k - 1]
        if DELTA: cur_d = pw[:, k - 1, [0, 1, 3]]
        t0 += k - 1
    return pred.numpy()

x_np, m_np = X.numpy(), M.numpy()
def yaw(a): return np.arctan2(a[..., 4], a[..., 3])
rows = {k: dict(disp_err=[], hold=[], true_disp=[], yaw_err=[], joint=[], pf=[], tf=[]) for k in KS}
for s in range(args.samples):
    torch.manual_seed(s); P = rollout()
    for k in KS:
        ok = m_np[:, k]
        dp = P[ok, k, :2] - x_np[ok, 0, :2]; dt_ = x_np[ok, k, :2] - x_np[ok, 0, :2]
        y0 = yaw(x_np[ok, 0]); c, sn = np.cos(y0), np.sin(y0)
        rows[k]["disp_err"] += list(np.linalg.norm(dp - dt_, axis=1)); rows[k]["hold"] += list(np.linalg.norm(dt_, axis=1))
        rows[k]["true_disp"] += list(np.linalg.norm(dt_, axis=1))
        rows[k]["yaw_err"] += list(np.degrees(np.abs(np.angle(np.exp(1j * (yaw(P[ok, k]) - yaw(x_np[ok, k])))))))
        rows[k]["joint"] += list(np.degrees(np.sqrt(((P[ok, k, 14:26] - x_np[ok, k, 14:26]) ** 2).mean(1))))
        rows[k]["pf"] += list(c * dp[:, 0] + sn * dp[:, 1]); rows[k]["tf"] += list(c * dt_[:, 0] + sn * dt_[:, 1])
out = {}
for k, v in rows.items():
    a = {q: np.array(w) for q, w in v.items()}
    out[f"{(k + 1) * 0.02:.1f}s"] = dict(n=int(len(a["disp_err"])), disp_err_cm_median=float(np.median(a["disp_err"]) * 100),
        true_disp_cm_median=float(np.median(a["true_disp"]) * 100), ratio_err_to_true=float(np.median(a["disp_err"]) / max(np.median(a["true_disp"]), 1e-6)),
        yaw_err_deg_median=float(np.median(a["yaw_err"])), joint_err_deg_median=float(np.median(a["joint"])),
        forward_corr=float(np.corrcoef(a["pf"], a["tf"])[0, 1]), forward_slope=float(np.polyfit(a["tf"], a["pf"], 1)[0]))
res = dict(checkpoint=str(args.ckpt), val_episodes=len(val), windows=len(X), starts_s=[t * 0.02 for t in STARTS], samples=args.samples, horizons=out)
Path(__file__).with_name("tt_horizon.json").write_text(json.dumps(res, indent=1))
for k, v in out.items(): print(k, {q: round(w, 3) for q, w in v.items()})
