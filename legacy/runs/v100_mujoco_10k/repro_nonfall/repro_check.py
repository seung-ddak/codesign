"""H64 TT 재현 확인: 넘어지지 않은 / 넘어진 검증 에피소드.

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

p = argparse.ArgumentParser()
p.add_argument("--ckpt", type=Path, required=True)
p.add_argument("--data", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--n", type=int, default=24)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(args.seed)
torch.set_num_threads(2)

ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
cfg = ck["args"]
H, clip = cfg["horizon"], cfg["clip"]
net = make_net(cfg)
net.load_state_dict(ck["ema_net"] if ck.get("ema_net") is not None else ck["net"])
net.eval().requires_grad_(False)
mu, sd = torch.as_tensor(ck["mu"]).float(), torch.as_tensor(ck["sd"]).float()
beta, abar = schedule(cfg["diff_steps"], "cpu")
emu, esd = (0.5, 0.5) if cfg["norm"] == "minmax" else (0.5, 0.25)
EP = 500

# 검증 에피소드 인덱스 -> (파일, 행). 학습 때 순서: e0000 v0, e0000 v1, e0001 v0 ... 각 100 행
files = []
for d in sorted(args.data.glob("e*")):
    files += [d / "variant_0.npz", d / "variant_1.npz"]
assert len(files) == 400, len(files)
val = np.asarray(ck["val"])
cache = {}


def episode(i):
    f, r = files[i // 100], i % 100
    if f not in cache:
        cache.clear()
        cache[f] = dict(np.load(f))
    d = cache[f]
    L = int(d["valid"][r].sum())
    x = np.zeros((EP, X_DIM), np.float32)
    x[:L, :S_DIM] = features(d["qpos"][r, :L], d["qvel"][r, :L], d["contact"][r, :L])
    x[:L, S_DIM:] = d["target_q"][r, :L]
    m = np.zeros(EP, bool)
    m[:L] = True
    return x, m, d["E"].astype(np.float32), bool(d["fell"][r]), str(d["motions"][r]), f"{f.parent.name}/{f.stem}#{r}"


fell_flag = {}
for i in val:
    f, r = files[i // 100], i % 100
    if f not in cache:
        cache.clear()
        cache[f] = dict(np.load(f))
    fell_flag[int(i)] = bool(cache[f]["fell"][r])
rng = np.random.default_rng(args.seed)
nf = rng.choice([i for i in fell_flag if not fell_flag[i]], args.n, replace=False)
fa = rng.choice([i for i in fell_flag if fell_flag[i]], args.n, replace=False)
pick = sorted(nf.tolist()) + sorted(fa.tolist())
eps = [episode(i) for i in pick]
X = torch.tensor(np.stack([e[0] for e in eps]))
M = torch.tensor(np.stack([e[1] for e in eps]))
E = torch.tensor(np.stack([e[2] for e in eps]))
En = (E - emu) / esd
FELL = np.array([e[3] for e in eps])
ar = torch.arange(H)
DELTA = cfg.get("pose", "abs") == "delta"


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


DX = deltas(X) if DELTA else None


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


def rollout(mode):
    """창 H 를 H-1 씩 이어 붙여 500 step 생성. mode 1: A 실제값 고정, mode 2: S·A 생성."""
    n = len(X)
    pred = X.clone()
    pred[:, 1:, :S_DIM] = 0
    if mode == 2:
        pred[:, :, S_DIM:] = 0
    cur = X[:, 0].clone()
    cur_d = DX[:, 0].clone() if DELTA else None
    t0 = 0
    while t0 < EP - 1:
        t = (t0 + ar).clamp(max=EP - 1)
        valid = (t0 + ar) < EP
        m = valid[None].expand(n, H) if mode == 2 else valid[None] & M[:, t]
        win = X[:, t].clone()
        win[:, 0, :S_DIM] = cur[:, :S_DIM]
        if mode == 2:
            win[:, 1:, :S_DIM] = 0
            win[:, :, S_DIM:] = 0
        xw = rebase(win)
        if DELTA:
            xw = set_delta(xw, DX[:, t])
            xw[:, 0, [0, 1, 3]] = cur_d
        xw = (xw - mu) / sd * m[..., None]
        kx, ke = condition_masks(torch.full((n,), mode, dtype=torch.long), m)
        if mode == 2:  # s0 의 이전 action 도 생성값을 쓰지 않는다: s0 상태만 고정
            kx[:, 0, S_DIM:] = 0
        px = ddim(xw, En, kx, ke, ~m)
        pw = px * sd + mu
        ep = integrate(pw, cur) if DELTA else unbase(pw, cur)
        k = min(H, EP - t0)
        pred[:, t0 + 1:t0 + k, :S_DIM] = ep[:, 1:k, :S_DIM]
        if mode == 2:
            pred[:, t0:t0 + k, S_DIM:] = ep[:, 0:k, S_DIM:]
        cur = ep[:, k - 1]
        if DELTA:
            cur_d = pw[:, k - 1, [0, 1, 3]]
        t0 += k - 1
    return pred.numpy()


Q, POS = slice(14, 26), slice(0, 2)
x_np, m_np = X.numpy(), M.numpy()


def metrics(pred, g):
    out = {}
    for t in (10, 50, 100, 250, 499):
        ok = m_np[g, t]
        if ok.sum() == 0:
            continue
        pp, tt = pred[g][ok, t], x_np[g][ok, t]
        hold = x_np[g][ok, 0]
        out[f"t{t}"] = dict(n=int(ok.sum()),
                            pos_err_cm=float(np.linalg.norm(pp[:, POS] - tt[:, POS], axis=-1).mean() * 100),
                            pos_hold_cm=float(np.linalg.norm(hold[:, POS] - tt[:, POS], axis=-1).mean() * 100),
                            joint_err_deg=float(np.degrees(np.sqrt(((pp[:, Q] - tt[:, Q]) ** 2).mean()))),
                            joint_hold_deg=float(np.degrees(np.sqrt(((hold[:, Q] - tt[:, Q]) ** 2).mean()))))
    mm = m_np[g]
    out["all"] = dict(pos_err_cm=float(np.linalg.norm(pred[g][..., POS] - x_np[g][..., POS], axis=-1)[mm].mean() * 100),
                      pos_hold_cm=float(np.linalg.norm(x_np[g][:, :1, POS] - x_np[g][..., POS], axis=-1)[mm].mean() * 100),
                      joint_err_deg=float(np.degrees(np.sqrt(((pred[g][..., Q] - x_np[g][..., Q]) ** 2)[mm].mean()))),
                      joint_hold_deg=float(np.degrees(np.sqrt(((x_np[g][:, :1, Q] - x_np[g][..., Q]) ** 2)[mm].mean()))))
    return out


t_start = time.time()
res = dict(checkpoint=str(args.ckpt), seed=args.seed, n_per_group=args.n, horizon=H, diff_steps=cfg["diff_steps"],
           episodes=[dict(id=int(i), src=e[5], motion=e[4], fell_in_data=e[3], steps=int(e[1].sum())) for i, e in zip(pick, eps)])
groups = {"nonfall": ~FELL, "fall": FELL}

p1 = rollout(1)
res["A_dynamics"] = {k: metrics(p1, g) for k, g in groups.items()}
print("A done", time.time() - t_start, flush=True)

p2 = rollout(2)
print("B gen done", time.time() - t_start, flush=True)
rows = []
for j in range(len(X)):
    e = E[j].numpy()
    _, _, meta_gen = replay(e, p2[j])
    tok = x_np[j].copy()
    L = int(m_np[j].sum())
    tok[L:, S_DIM:] = tok[L - 1, S_DIM:]  # 넘어진 뒤 구간은 마지막 action 유지
    _, _, meta_real = replay(e, tok)
    act_gap = float(np.degrees(np.sqrt(((p2[j, :L, S_DIM:] - x_np[j, :L, S_DIM:]) ** 2).mean())))
    rows.append(dict(id=int(pick[j]), fell_in_data=bool(FELL[j]), data_seconds=L * 0.02,
                     real_A_seconds=meta_real["valid_seconds"], real_A_reason=meta_real["termination_reason"],
                     gen_A_seconds=meta_gen["valid_seconds"], gen_A_reason=meta_gen["termination_reason"],
                     gen_vs_real_action_deg=act_gap, gen_forward_m=meta_gen.get("forward_m"),
                     real_forward_m=float(x_np[j, L - 1, 0] - x_np[j, 0, 0])))
res["B_generation_replay"] = rows
res["B_dynamics_of_generated_S"] = {k: metrics(p2, g) for k, g in groups.items()}
res["seconds"] = time.time() - t_start
json.dump(res, open(args.out / "result.json", "w"), indent=1, ensure_ascii=False)
np.savez_compressed(args.out / "samples.npz", true=x_np, mask=m_np, mode1=p1, mode2=p2, E=E.numpy(), fell=FELL, ids=np.array(pick))
for k in groups:
    r = [x for x in rows if x["fell_in_data"] == (k == "fall")]
    print(k, "real-A survive", sum(x["real_A_reason"] == "completed_horizon" for x in r), "/", len(r),
          "gen-A survive", sum(x["gen_A_reason"] == "completed_horizon" for x in r),
          "gen mean sec", np.mean([x["gen_A_seconds"] for x in r]))
print("DONE", res["seconds"])
