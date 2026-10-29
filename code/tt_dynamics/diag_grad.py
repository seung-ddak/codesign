"""guidance gradient 크기와 E 궤적 검증 (TT 만, MJX 없음). generate_task.py 첫 window 와 같은 계산을 step 마다 기록한다.

기록: denoising step k 마다 |∇S R|·|∇A R|·|∇E R| (token 당 평균 L2), 그 step 의 Ê0 와 가장 가까운 학습 형상·거리.
자체 점검: 마지막 E 가 같은 seed 의 generate_task 결과 (--check summary.json) 와 같아야 한다.

usage (codesign/ 에서):
  CUDA_VISIBLE_DEVICES=1 ../transformer-transformer/.venv/bin/python code/tt_dynamics/diag_grad.py \
      --ckpt runs/pipeline_v1/tt/h32/try6 --check runs/pipeline_v1/task/track/try4/summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rollout_data import E_DIM, S_DIM, X_DIM
from task_reward import TrackReward
from tt_model import Net, condition_masks, rest_s0, schedule, unbase

CODESIGN = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", type=Path, required=True)
parser.add_argument("--check", type=Path, help="같은 설정 generate_task summary.json (rest-s0, circle)")
parser.add_argument("--scales", type=float, nargs="+", default=[1.0, 3.0, 10.0])
parser.add_argument("--candidates", type=int, default=8)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
dev = "cuda"

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
reward = TrackReward()
SHAPES = {n: np.load(CODESIGN / f"runs/pipeline_v1/rollouts/{n}/seq_seed0.npz")["E"] for n in ["e000", "e001", "e002"]}


def nearest(e):
    d = {n: np.linalg.norm(e - v, axis=-1) for n, v in SHAPES.items()}
    names = np.array(list(d))
    D = np.stack(list(d.values()), -1)
    return names[D.argmin(-1)], D.min(-1)


@torch.no_grad()
def first_window(scale, n):
    """generate_task.generate 의 첫 window (rest-s0, mode 4) 를 그대로 재현하며 step 별 기록."""
    torch.manual_seed(args.seed)
    cur = rest_s0().to(dev).expand(n, S_DIM).clone()
    m = torch.ones(n, H, dtype=torch.bool, device=dev)
    x0 = torch.zeros(n, H, X_DIM, device=dev)
    x0[:, 0, :S_DIM] = (cur - mu[:S_DIM]) / sd[:S_DIM]
    kx, ke = condition_masks(torch.full((n,), 4, device=dev), m)
    x0, e0, pad = x0 * m[..., None], torch.zeros(n, E_DIM, device=dev), ~m
    x, e = torch.randn_like(x0), torch.randn_like(e0)
    log = []
    for k in reversed(range(len(beta))):
        ab = abar[k]
        ab_prev = abar[k - 1] if k > 0 else torch.ones((), device=dev)
        kk = torch.full((n,), k, device=dev, dtype=torch.float32)
        with torch.enable_grad():
            xr, er = x.detach().requires_grad_(), e.detach().requires_grad_()
            ex, ee = net(torch.where(kx > 0, x0, xr), torch.where(ke > 0, e0, er), kx, ke, kk, pad)
            x0h = torch.where(kx > 0, x0, ((xr - (1 - ab).sqrt() * ex) / ab.sqrt()).clamp(-CLIP, CLIP))
            R, _ = reward(unbase(x0h * sd + mu, cur), m)
            gx, ge = torch.autograd.grad(R.sum(), [xr, er])
        e0h_pre = ((e - (1 - ab).sqrt() * ee.detach()) / ab.sqrt()).clamp(-CLIP, CLIP)
        ex, ee = (ex - scale * (1 - ab).sqrt() * gx).detach(), (ee - scale * (1 - ab).sqrt() * ge).detach()
        x0h = ((x - (1 - ab).sqrt() * ex) / ab.sqrt()).clamp(-CLIP, CLIP)
        e0h = ((e - (1 - ab).sqrt() * ee) / ab.sqrt()).clamp(-CLIP, CLIP)
        nm, dist = nearest((e0h * esd + emu).clamp(0, 1).cpu().numpy())
        log.append(dict(k=k, w=float((1 - ab).sqrt() * scale), gS=gx[..., :S_DIM].norm(dim=-1).mean().item(),
                        gA=gx[..., S_DIM:].norm(dim=-1).mean().item(), gE=ge.norm(dim=-1).mean().item(),
                        dE_guid=((e0h - e0h_pre) * esd).norm(dim=-1).mean().item(), shape=nm.tolist(), dist=dist.round(3).tolist()))
        epx, epe = (x - ab.sqrt() * x0h) / (1 - ab).sqrt(), (e - ab.sqrt() * e0h) / (1 - ab).sqrt()
        x, e = ab_prev.sqrt() * x0h + (1 - ab_prev).sqrt() * epx, ab_prev.sqrt() * e0h + (1 - ab_prev).sqrt() * epe
    return (e * esd + emu).clamp(0, 1).cpu().numpy(), log


out = {"diff_steps": len(beta), "abar": abar.tolist()}
for scale in args.scales:
    E, log = first_window(scale, args.candidates)
    if args.check:
        ref = np.array([r["E"] for r in json.load(open(args.check))["runs"][f"scale_{scale:g}"]])
        err = np.abs(E.round(3) - ref).max()
        print(f"scale {scale:g} 자체 점검: generate_task E 와 최대 차 {err:.4f}", "PASS" if err < 2e-3 else "FAIL")
    for r in log:
        print(f"scale {scale:g} k={r['k']} 가중 {r['w']:.2f} | |∇S| {r['gS']:.3g} |∇A| {r['gA']:.3g} |∇E| {r['gE']:.3g} | guidance 로 바뀐 Ê0 {r['dE_guid']:.3g} | "
              f"형상 {''.join(s[-1] for s in r['shape'])}", flush=True)
    out[f"scale_{scale:g}"] = log
dst = args.ckpt / "diag_grad.json"
json.dump(out, open(dst, "w"), indent=1)
print("->", dst)
