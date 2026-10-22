"""TT joint dynamics: rollout npz 를 직접 읽어 p(E, S, A | s0) 를 diffusion transformer 로 학습한다.

usage (codesign/ 에서):
  CUDA_VISIBLE_DEVICES=0 ../transformer-transformer/.venv/bin/python code/tt_dynamics/train_tt.py \
      --out runs/pipeline_v1/tt/h500_try1
  평가만: ... --out <run> --eval-only

토큰: E 토큰 1 개 (연속 치수 13) + step 토큰 H 개 ([S_t, A_t], t = 0..H-1)
  S_t (42) = 시작 heading frame 기준 xy 위치 2, 몸통 높이 1, 시작 대비 yaw cos·sin 2, body frame 중력 3,
             heading frame 선속도 3, body 각속도 3, 관절각 12, 관절속도 12, 발 접지 4
  A_t (12) = target_q (실제 PD 목표 관절각)
  위치·yaw 는 window 첫 state 기준. 낙상 이후 (valid = 0) 와 padding 은 attention·loss 에서 뺀다.
학습 조건 (배치마다 무작위): joint = s0 만 / dynamics = E·s0·A / given_E = E·s0 / free = 없음 (--modes 에 3). 알려진 값은 매 step 원래 값으로 고정 (inpainting).
평가: 검증 episode 에서 dynamics 조건으로 S 를 생성해 실제 MJX S 와 비교 (1·10·50 step, 전체).
  horizon < 500: 마지막 예측 state 를 다음 window 의 s0 로 두고 이어 붙인다 (window 당 H-1 step 전진).
"""

import argparse
import glob
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rollout_data import A_DIM, E_DIM, S_DIM, X_DIM, features, load, source_group_keys  # noqa: F401
from tt_model import CONDITIONING_VERSION, Net, condition_masks, rebase, schedule, unbase
Q = slice(14, 26)  # S 안의 관절각 (pos 2, z 1, yaw 2, grav 3, vel 3, angvel 3 다음)
POS = slice(0, 2)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--data", nargs="+", default=sorted(glob.glob("runs/pipeline_v1/rollouts/e00[0-2]/*_seed0.npz")))
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--horizon", type=int, default=500, help="학습 window 길이 (step). < 500 이면 무작위 구간, 평가는 window 를 이어 붙여 500 step")
parser.add_argument("--steps", type=int, default=20000)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-4)
parser.add_argument("--d-model", type=int, default=256)
parser.add_argument("--layers", type=int, default=6)
parser.add_argument("--diff-steps", type=int, default=100)
parser.add_argument("--val-frac", type=float, default=0.1)
parser.add_argument("--split-key", help="NPZ row key with fixed train/val/test labels; test rows are excluded from training")
parser.add_argument("--norm-from", type=Path, help="reuse another checkpoint's train-only x normalization")
parser.add_argument("--eval-episodes", type=int, default=64)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--eval-only", action="store_true")
parser.add_argument("--norm", choices=("zscore", "minmax"), default="zscore", help="minmax = 학습 데이터 범위를 [-1,1] 로 (TT 원본)")
parser.add_argument("--clip", type=float, default=10.0, help="정규화 공간 x0_hat clip (TT 원본 clip_sample = 1)")
parser.add_argument("--stride", type=int, default=0, help="평가 시 window 당 전진 step (0 = horizon-1, 겹침 없음)")
parser.add_argument("--pose", choices=("abs", "delta"), default="abs", help="delta = S 의 위치·yaw 4 칸을 직전 step 대비 변화량 (dx, dy: 직전 heading frame, dyaw, 0) 으로")
parser.add_argument("--modes", type=int, nargs="+", choices=(0, 1, 2, 3, 4, 5), default=[0, 1, 2], help="학습 조건: 0 joint (s0), 1 dynamics (E·s0·A), 2 given_E (E·s0), 3 free (없음), 4 anchor (s0 XY/yaw만 고정), 5 fixed-E anchor (E·s0 XY/yaw 고정)")
parser.add_argument("--diag", action="store_true", help="이어 붙이지 않은 단일 window 예측의 몸통 yaw·속도 오차를 모션 종류별로 (eval-only)")
parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm", help="평가 샘플링. ddim = 결정적 (eta 0)")
args = parser.parse_args()
args.conditioning_version = CONDITIONING_VERSION
dev = "cuda"
torch.manual_seed(args.seed)


EP = 500  # episode 최대 step (10 s)



@torch.no_grad()
def sample(net, x0, e0, kx, ke, pad, beta, abar):
    x, e = torch.randn_like(x0), torch.randn_like(e0)
    for k in reversed(range(len(beta))):
        x, e = torch.where(kx > 0, x0, x), torch.where(ke > 0, e0, e)
        kk = torch.full((len(x),), k, device=dev, dtype=torch.float32)
        ex, ee = net(torch.where(kx > 0, x0, x), torch.where(ke > 0, e0, e), kx, ke, kk, pad)
        # posterior q(x_{k-1} | x_k, x0_hat), x0_hat 는 정규화 공간에서 clip (마지막 step β≈1 의 eps 오차 증폭 방지)
        ab, ab_prev = abar[k], (abar[k - 1] if k > 0 else torch.ones((), device=dev))
        c0, ck = ab_prev.sqrt() * beta[k] / (1 - ab), (1 - beta[k]).sqrt() * (1 - ab_prev) / (1 - ab)
        x0h = ((x - (1 - ab).sqrt() * ex) / ab.sqrt()).clamp(-args.clip, args.clip)
        e0h = ((e - (1 - ab).sqrt() * ee) / ab.sqrt()).clamp(-args.clip, args.clip)
        if args.sampler == "ddim":  # 결정적: clip 한 x0_hat 에 맞춘 eps 로 x_{k-1} 을 바로 계산
            epx, epe = (x - ab.sqrt() * x0h) / (1 - ab).sqrt(), (e - ab.sqrt() * e0h) / (1 - ab).sqrt()
            x, e = ab_prev.sqrt() * x0h + (1 - ab_prev).sqrt() * epx, ab_prev.sqrt() * e0h + (1 - ab_prev).sqrt() * epe
            continue
        x, e = c0 * x0h + ck * x, c0 * e0h + ck * e
        if k > 0:
            sig = (beta[k] * (1 - ab_prev) / (1 - ab)).sqrt()
            x, e = x + sig * torch.randn_like(x), e + sig * torch.randn_like(e)
    return torch.where(kx > 0, x0, x), torch.where(ke > 0, e0, e)




X, E, M, src, CL = load(args.data, EP)
H = args.horizon
rng = np.random.default_rng(args.seed)
# Split by physical source episode across morphologies, not the file path.
# group_id (when supplied) also groups counterfactual actions/morphology.
groups = []
for path in args.data:
    with np.load(path) as z:
        rows = [b for b in range(len(z["valid"])) if min(int(z["valid"][b].sum()), EP) >= 2]
        keys = source_group_keys(path, z, length=len(z["valid"]))
        groups.extend(keys[b] for b in rows)
assert len(groups) == len(X)
unique, inverse = np.unique(groups, return_inverse=True)
if args.split_key:
    labels = []
    for path in args.data:
        with np.load(path) as z:
            if args.split_key not in z:
                raise KeyError(f"{args.split_key} missing in {path}")
            labels.extend(z[args.split_key][b] for b in range(len(z["valid"])) if min(int(z["valid"][b].sum()), EP) >= 2)
    labels = np.asarray(labels)
    assert len(labels) == len(X) and set(labels) <= {"train", "val", "test"}
    tr, val = np.flatnonzero(labels == "train"), np.flatnonzero(labels == "val")
else:
    assert len(unique) >= 2, "at least two independent starts are needed for train/validation"
    perm = rng.permutation(len(unique))
    n_val_groups = min(len(unique) - 1, max(1, int(len(unique) * args.val_frac)))
    val, tr = np.where(np.isin(inverse, perm[:n_val_groups]))[0], np.where(np.isin(inverse, perm[n_val_groups:]))[0]
assert len(tr) and len(val), "train and val must both be nonempty"
n_val = len(val)
assert not np.intersect1d(inverse[val], inverse[tr]).size
# Count distinct start feature vectors, not merely the number of episodes.
# Raw S0 includes morphology-dependent joint/base poses, but this statistic
# does not prove real-world start diversity.
distinct_s0 = len(np.unique(np.round(X[:, 0, 2:38], decimals=5), axis=0))
split_meta = {
    "episodes": int(len(X)),
    "train_episodes": int(len(tr)),
    "validation_episodes": int(n_val),
    "independent_source_groups": int(len(unique)),
    "train_source_groups": int(len(np.unique(inverse[tr]))),
    "validation_source_groups": int(len(np.unique(inverse[val]))),
    "distinct_s0_feature_vectors_rounded_1e-5": int(distinct_s0),
    "split_group_contract": "cross-morphology source_episode or group_id; see rollout_data.source_group_keys",
    "validation_loss_scope": "all validation episodes, batched; random-window/noise, fixed evaluation RNG seed",
}
print("split metadata:", split_meta, flush=True)
Xt, Mt = torch.tensor(X, device=dev), torch.tensor(M, device=dev)
Lt = Mt.sum(1)


def deltas(x):
    """episode 기준 위치·yaw [B,T,X] -> 직전 step 대비 (dx, dy, dyaw) [B,T,3]. t=0 은 0."""
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
    """delta 표현 window [B,H,X] 를 start (episode 기준 [B,X]) 부터 적분 -> episode 기준 위치·yaw."""
    out = w.clone()
    p, yaw = start[:, 0:2].clone(), torch.atan2(start[:, 4], start[:, 3])
    out[:, 0, 0:2], out[:, 0, 3], out[:, 0, 4] = p, torch.cos(yaw), torch.sin(yaw)
    for i in range(1, w.shape[1]):
        c, s = torch.cos(yaw), torch.sin(yaw)
        p = p + torch.stack([c * w[:, i, 0] - s * w[:, i, 1], s * w[:, i, 0] + c * w[:, i, 1]], -1)
        yaw = yaw + w[:, i, 3]
        out[:, i, 0:2], out[:, i, 3], out[:, i, 4] = p, torch.cos(yaw), torch.sin(yaw)
    return out


Dt = deltas(Xt) if args.pose == "delta" else None
ar = torch.arange(H, device=dev)


def window(idx, gen=None):
    """episode idx 마다 무작위 시작 (가능하면 window 전체가 유효한 구간) -> 기준 변환한 x [n,H,X], mask [n,H]."""
    L = Lt[idx]
    hi = (L - torch.clamp(L, max=H)).float()
    start = (torch.rand(len(idx), device=dev, generator=gen) * (hi + 1)).long().clamp(max=hi.long())
    t = start[:, None] + ar
    m = (t < L[:, None])
    x = rebase(Xt[idx[:, None], t.clamp(max=EP - 1)])
    if Dt is not None:
        x = set_delta(x, Dt[idx[:, None], t.clamp(max=EP - 1)])
    return x * m[..., None], m


g = torch.Generator(device=dev).manual_seed(args.seed)
trt = torch.tensor(tr, device=dev)
if args.norm_from:
    norm_ck = torch.load(args.norm_from, map_location="cpu", weights_only=False)
    if norm_ck["args"]["norm"] != args.norm or norm_ck["args"]["horizon"] != H:
        raise ValueError("normalization checkpoint has different norm or horizon")
    mu, sd = norm_ck["mu"], norm_ck["sd"]
else:
    xs, ms = zip(*[window(trt[torch.randint(0, len(tr), (2048,), device=dev, generator=g)], g) for _ in range(10)])
    flat = torch.cat([x[m] for x, m in zip(xs, ms)]).cpu().numpy()
    if args.norm == "minmax":
        lo, hi = flat.min(0), flat.max(0)
        mu, sd = (hi + lo) / 2, (hi - lo) / 2 + 1e-3
    else:
        mu, sd = flat.mean(0), flat.std(0) + 1e-3
emu, esd = (0.5, 0.5) if args.norm == "minmax" else (0.5, 0.25)
mu_t, sd_t = torch.tensor(mu, device=dev), torch.tensor(sd, device=dev)
En = torch.tensor((E - emu) / esd, device=dev)
args.out.mkdir(parents=True, exist_ok=True)
beta, abar = schedule(args.diff_steps, dev)
net = Net(H, args.d_model, args.layers).to(dev)
ckpt_path = args.out / "checkpoint.pt"
print(f"episodes={len(X)} (train {len(tr)}, val {n_val}) valid steps={int(M.sum())} horizon={H} sources={sorted(set(src))}")
print(f"params={sum(p.numel() for p in net.parameters()) / 1e6:.2f}M")

if not args.eval_only:
    json.dump({**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "episodes": len(X), "train": len(tr), "val": n_val,
               "data_split_audit": split_meta}, open(args.out / "config.json", "w"), indent=1)
    (args.out / "split_audit.json").write_text(json.dumps(split_meta, indent=2) + "\\n")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    log = open(args.out / "progress.jsonl", "w")
    valt, t0 = torch.tensor(val, device=dev), time.time()

    def loss_on(idx, gen=None):
        x, m = window(idx, gen)
        x0, e0 = (x - mu_t) / sd_t * m[..., None], En[idx]
        n = len(idx)
        modes = torch.tensor(args.modes, device=dev)
        mode = modes[torch.randint(0, len(modes), (n,), device=dev, generator=gen)]
        kx, ke = condition_masks(mode, m)
        k = torch.randint(0, args.diff_steps, (n,), device=dev, generator=gen)
        ab = abar[k]
        nx, ne = torch.randn(x0.shape, device=dev, generator=gen), torch.randn(e0.shape, device=dev, generator=gen)
        xt = ab.sqrt()[:, None, None] * x0 + (1 - ab).sqrt()[:, None, None] * nx
        et = ab.sqrt()[:, None] * e0 + (1 - ab).sqrt()[:, None] * ne
        px, pe = net(torch.where(kx > 0, x0, xt), torch.where(ke > 0, e0, et), kx, ke, k.float(), ~m)
        wx = (1 - kx) * m[..., None]
        return ((px - nx) ** 2 * wx).sum() / wx.sum() + (((pe - ne) ** 2) * (1 - ke)).mean()

    for step in range(1, args.steps + 1):
        loss = loss_on(trt[torch.randint(0, len(tr), (args.batch,), device=dev)])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == args.steps:
            # Evaluate ALL validation episodes, not just valt[:batch].
            # Each batch receives deterministic random windows/noise under
            # one generator per checkpoint evaluation; weight by row count.
            with torch.no_grad():
                val_gen = torch.Generator(device=dev).manual_seed(1)
                total_v = 0.0
                for j in range(0, len(valt), args.batch):
                    vb = valt[j:j + args.batch]
                    total_v += float(loss_on(vb, val_gen)) * len(vb)
                vl = total_v / len(valt)
            rec = dict(step=step, loss=loss.item(), val_loss=vl,
                       val_episodes_evaluated=len(valt), val_batches=(len(valt) + args.batch - 1) // args.batch,
                       lr=sched.get_last_lr()[0], min=(time.time() - t0) / 60)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
    torch.save({"net": net.state_dict(), "mu": mu, "sd": sd, "val": val, "args": vars(args)}, ckpt_path)

ck = torch.load(ckpt_path, weights_only=False)
net.load_state_dict(ck["net"])
net.eval()
# The historical post-training evaluation requires Mode 1 (E+s0+A -> S).
# With joint-only Modes 2+4, it would test an UNTRAINED conditioning mode
# and mislabel an unconditional action sample as dynamics reconstruction.
# Validate joint-only checkpoints with task generation and MJX instead.
if 1 not in ck["args"]["modes"]:
    print("Joint-only checkpoint: Modes 2/4; skipping Mode-1 dynamics diagnostics. "
          "Check val_loss in progress.jsonl, then run FAST joint generation + MJX.", flush=True)
    raise SystemExit(0)
if args.diag:
    # 검증 episode 마다 시작점 8 개 (window 전체 유효), 실제 s0 에서 window 하나만 예측
    torch.manual_seed(args.seed)
    vt = torch.tensor(val, device=dev).repeat_interleave(8)
    Lv = Lt[vt]
    ok = Lv >= H
    vt, Lv = vt[ok], Lv[ok]
    st = (torch.rand(len(vt), device=dev) * (Lv - H + 1).float()).long()
    t = st[:, None] + ar
    P, T = [], []
    for i in range(0, len(vt), 256):
        idx, tt = vt[i:i + 256], t[i:i + 256]
        x = rebase(Xt[idx[:, None], tt])
        if Dt is not None:
            x = set_delta(x, Dt[idx[:, None], tt])
        m = torch.ones(len(idx), H, dtype=torch.bool, device=dev)
        kx, ke = condition_masks(torch.ones(len(idx), dtype=torch.long, device=dev), m)
        px, _ = sample(net, (x - mu_t) / sd_t, En[idx], kx, ke, ~m, beta, abar)
        pw = px * sd_t + mu_t
        start = torch.zeros(len(idx), X_DIM, device=dev)
        start[:, 3] = 1  # window 기준 원점
        pw = integrate(pw, start) if Dt is not None else pw
        P.append(pw.cpu().numpy()), T.append(rebase(Xt[idx[:, None], tt]).cpu().numpy())
    P, T = np.concatenate(P), np.concatenate(T)
    lab = CL[vt.cpu().numpy(), st.cpu().numpy()]
    kind = np.array(["정지" if ("squat" in c or "tilt" in c) else "회전" if ("CCW" in c or "CW" in c) else "직진·횡이동" for c in lab])
    yaw = lambda a: np.arctan2(a[..., 4], a[..., 3])
    wrap = lambda a: np.angle(np.exp(1j * a))
    k = H - 1
    out = {}
    for name in ("정지", "직진·횡이동", "회전", "전체"):
        g = kind == name if name != "전체" else np.ones(len(kind), bool)
        p, tr_, = P[g], T[g]
        dy_p, dy_t = np.degrees(wrap(yaw(p[:, k]))), np.degrees(wrap(yaw(tr_[:, k])))
        out[name] = dict(
            n=int(g.sum()),
            yaw_err_deg=float(np.abs(wrap(np.radians(dy_p - dy_t))).mean() * 180 / np.pi), yaw_hold_deg=float(np.abs(dy_t).mean()),
            yaw_bias_deg=float((dy_p - dy_t).mean()), yaw_slope=float(np.polyfit(dy_t, dy_p, 1)[0]) if g.sum() > 2 else None,
            yaw_corr=float(np.corrcoef(dy_t, dy_p)[0, 1]) if g.sum() > 2 else None,
            vel_err=float(np.linalg.norm(p[:, 1:, 8:10] - tr_[:, 1:, 8:10], axis=-1).mean()),
            vel_hold=float(np.linalg.norm(tr_[:, :1, 8:10] - tr_[:, 1:, 8:10], axis=-1).mean()),
            wz_err=float(np.abs(p[:, 1:, 13] - tr_[:, 1:, 13]).mean()), wz_hold=float(np.abs(tr_[:, :1, 13] - tr_[:, 1:, 13]).mean()),
            pos_err_cm=float(np.linalg.norm(p[:, k, :2] - tr_[:, k, :2], axis=-1).mean() * 100),
            pos_hold_cm=float(np.linalg.norm(tr_[:, k, :2], axis=-1).mean() * 100),
            joint_err_deg=float(np.degrees(np.sqrt(((p[:, 1:, Q] - tr_[:, 1:, Q]) ** 2).mean()))),
        )
    # 데이터 자체 점검: 실제 yaw 변화 vs body 각속도 z 적분 (평지에서 거의 같아야 한다)
    integ = np.degrees(T[:, 1:k + 1, 13].sum(1) * 0.02)
    out["check_true_yaw_vs_wz_integral"] = dict(corr=float(np.corrcoef(integ, np.degrees(wrap(yaw(T[:, k]))))[0, 1]),
                                               mean_abs_diff_deg=float(np.abs(integ - np.degrees(wrap(yaw(T[:, k])))).mean()))
    json.dump(out, open(args.out / f"diag_window{'_delta' if Dt is not None else ''}.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    raise SystemExit
# 평가: dynamics 조건 (E·s0·A 고정) 으로 S 생성 -> 실제 MJX S 와 비교. window 를 이어 붙여 episode 전체를 예측한다
ev = torch.tensor(val[:args.eval_episodes], device=dev)
true_t, mask_t = Xt[ev], Mt[ev]
pred_t = true_t.clone()
pred_t[:, 1:, :S_DIM] = 0
cur = true_t[:, 0].clone()
cur_d = Dt[ev, 0] if Dt is not None else None
t0 = 0
stride = args.stride or H - 1
assert 1 <= stride <= H - 1
torch.manual_seed(args.seed)
while t0 < EP - 1 and bool(mask_t[:, t0].any()):
    t = t0 + ar
    m = (t < EP) & mask_t[:, t.clamp(max=EP - 1)]
    win = true_t[:, t.clamp(max=EP - 1)].clone()
    win[:, 0, :S_DIM] = cur[:, :S_DIM]  # s0 = 직전 window 의 마지막 예측 state
    xw = rebase(win)
    if Dt is not None:  # s0 의 변화량 = 직전 window 의 마지막 예측 (첫 window 는 실제)
        xw = set_delta(xw, Dt[ev[:, None], t.clamp(max=EP - 1)])
        xw[:, 0, [0, 1, 3]] = cur_d
    xw = (xw - mu_t) / sd_t * m[..., None]
    kx, ke = condition_masks(torch.ones(len(ev), dtype=torch.long, device=dev), m)
    px, _ = sample(net, xw, En[ev], kx, ke, ~m, beta, abar)
    pw = px * sd_t + mu_t
    ep = integrate(pw, cur) if Dt is not None else unbase(pw, cur)
    n = min(stride + 1, EP - t0)  # window 앞쪽 stride step 만 쓰고 나머지는 버린다
    pred_t[:, t0 + 1:t0 + n, :S_DIM] = torch.where(m[:, 1:n, None], ep[:, 1:n, :S_DIM], pred_t[:, t0 + 1:t0 + n, :S_DIM])
    cur = ep[:, n - 1]
    if Dt is not None:
        cur_d = pw[:, n - 1, [0, 1, 3]]
    t0 += n - 1
pred, true, mask = pred_t.cpu().numpy(), true_t.cpu().numpy(), mask_t.cpu().numpy()
res = {}
for t in (1, 10, 50, 100, 250, 499):
    ok = mask[:, t]
    if ok.sum() == 0:
        continue
    res[f"t{t}"] = dict(n=int(ok.sum()),
                        joint_rmse_deg=float(np.degrees(np.sqrt(((pred[ok, t, Q] - true[ok, t, Q]) ** 2).mean()))),
                        pos_err_cm=float(np.linalg.norm(pred[ok, t, POS] - true[ok, t, POS], axis=-1).mean() * 100))
res["all"] = dict(joint_rmse_deg=float(np.degrees(np.sqrt(((pred[..., Q] - true[..., Q]) ** 2)[mask].mean()))),
                  pos_err_cm=float(np.linalg.norm(pred[..., POS] - true[..., POS], axis=-1)[mask].mean() * 100))
# 비교 기준: 모든 step 을 s0 로 둔 예측 (움직임 없음)
hold = np.repeat(true[:, :1], true.shape[1], 1)
res["baseline_hold_s0_all"] = dict(joint_rmse_deg=float(np.degrees(np.sqrt(((hold[..., Q] - true[..., Q]) ** 2)[mask].mean()))),
                                   pos_err_cm=float(np.linalg.norm(hold[..., POS] - true[..., POS], axis=-1)[mask].mean() * 100))
tag = ("" if args.sampler == "ddpm" else f"_{args.sampler}") + (f"_stride{args.stride}" if args.stride else "") + ("_delta" if args.pose == "delta" else "")
json.dump(res, open(args.out / f"eval_dynamics{tag}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
np.savez_compressed(args.out / f"eval_dynamics{tag}_samples.npz", pred=pred, true=true, mask=mask, src=src[val[:args.eval_episodes]])
