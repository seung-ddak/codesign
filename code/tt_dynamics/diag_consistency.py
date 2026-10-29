"""생성 궤적의 내부 일관성: 생성된 S 가 생성된 E·A 에 대해 TT 자신이 예측하는 S (dynamics 조건) 와 얼마나 다른가.

MJX 를 거치지 않으므로 TT dynamics 자체 오차는 빠지고, guidance 가 S 를 E·A 와 무관하게 옮긴 양만 남는다.
window (H step, H-1 간격) 마다 생성 궤적의 그 시점 state 를 s0 로 두고 (이어 붙이지 않음) E·s0·A 를 고정해 S 를 다시 생성한다.

usage (codesign/ 에서):
  CUDA_VISIBLE_DEVICES=1 ../transformer-transformer/.venv/bin/python code/tt_dynamics/diag_consistency.py \
      --ckpt runs/pipeline_v1/tt/h32/try4 --runs runs/pipeline_v1/task/track/try1 runs/pipeline_v1/task/track/try3
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rollout_data import S_DIM
from tt_model import Net, rebase, schedule

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", type=Path, required=True)
parser.add_argument("--runs", type=Path, nargs="+", required=True)
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
Q = slice(14, 26)


@torch.no_grad()
def dynamics(x0, e0):
    """E·s0·A 고정, S 만 DDIM (결정적) 으로 생성. x0 [B,H,X] 정규화 공간."""
    kx = torch.zeros_like(x0)
    kx[:, 0, :S_DIM], kx[:, :, S_DIM:] = 1, 1
    ke, pad = torch.ones_like(e0), torch.zeros(x0.shape[:2], dtype=torch.bool, device=dev)
    x = torch.randn_like(x0)
    for k in reversed(range(len(beta))):
        ab, ab_prev = abar[k], (abar[k - 1] if k > 0 else torch.ones((), device=dev))
        x = torch.where(kx > 0, x0, x)
        ex, _ = net(x, e0, kx, ke, torch.full((len(x),), k, device=dev, dtype=torch.float32), pad)
        x0h = ((x - (1 - ab).sqrt() * ex) / ab.sqrt()).clamp(-CLIP, CLIP)
        x = ab_prev.sqrt() * x0h + (1 - ab_prev).sqrt() * (x - ab.sqrt() * x0h) / (1 - ab).sqrt()
    return torch.where(kx > 0, x0, x)


def yaw(x):
    return torch.atan2(x[..., 4], x[..., 3])


out = {}
for run in args.runs:
    for f in sorted(run.glob("scale_*.npz")):
        torch.manual_seed(args.seed)
        z = np.load(f)
        pred, E = torch.tensor(z["pred"], device=dev), torch.tensor(z["E"], device=dev)
        n, EP = pred.shape[:2]
        starts = range(0, EP - H + 1, H - 1)
        w = torch.cat([rebase(pred[:, t:t + H]) for t in starts])  # [n·W,H,X] window 기준
        e = ((E - emu) / esd).repeat(len(starts), 1)
        dyn = dynamics((w - mu) / sd, e) * sd + mu
        dp = (dyn[..., 0:2] - w[..., 0:2]).norm(dim=-1) * 100
        dyaw = torch.remainder(yaw(dyn) - yaw(w) + math.pi, 2 * math.pi).sub(math.pi).abs().rad2deg()
        dq = (dyn[..., Q] - w[..., Q]).abs().rad2deg().mean(-1)
        key = f"{run.name}/{f.stem}"
        out[key] = dict(windows=len(w), pos_end_cm=dp[:, -1].mean().item(), yaw_end_deg=dyaw[:, -1].mean().item(),
                        joint_mean_deg=dq.mean().item(), pos_mean_cm=dp.mean().item())
        print(key, {k: round(v, 2) for k, v in out[key].items()}, flush=True)

dst = args.runs[-1] / "diag_consistency.json"
json.dump(out, open(dst, "w"), indent=1)
print("->", dst)
