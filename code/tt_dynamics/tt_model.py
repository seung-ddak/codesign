"""TT 모델 (diffusion transformer) 과 window 좌표 변환. train_tt.py 와 generate_task.py 가 같이 쓴다."""

import math

import torch
import torch.nn as nn

from rollout_data import E_DIM, S_DIM, X_DIM


CONDITIONING_VERSION = 4


# For Mode 4, constrain ONLY the canonical heading-frame anchor.
# A training window is rebased to position (0,0) and yaw (cos,sin)=(1,0);
# velocity, gravity, height, q, qdot and foot contact are all sampled from
# the MJX data distribution, not forced to artificial rest values.
ANCHOR_KNOWN = torch.zeros(S_DIM)
ANCHOR_KNOWN[[0, 1, 3, 4]] = 1
# Legacy name for older callers; same revised semantics and a new checkpoint version.
REST_KNOWN = ANCHOR_KNOWN


def anchor_s0():
    """Canonical first-frame position/yaw only. All other entries UNKNOWN."""
    s = torch.zeros(S_DIM)
    s[3] = 1.0
    return s


def rest_s0():
    """Backward-compatible name. Mode 4 no longer assumes stationary stance."""
    return anchor_s0()


def condition_masks(mode, valid):
    """Known-value masks: 0=s0, 1=E+s0+A, 2=E+s0, 3=free, 4=anchor (generate E), 5=E+anchor (fix E, generate s0/S/A)."""
    n, horizon = valid.shape
    kx = torch.zeros(n, horizon, X_DIM, device=valid.device)
    s0_known = torch.where(((mode == 4) | (mode == 5))[:, None], ANCHOR_KNOWN.to(valid.device), torch.ones(S_DIM, device=valid.device))
    kx[:, 0, :S_DIM] = (mode != 3).float()[:, None] * s0_known
    kx[:, :, S_DIM:] = (mode == 1).float()[:, None, None]
    ke = ((mode == 1) | (mode == 2) | (mode == 5)).float()[:, None].expand(n, E_DIM)
    return kx * valid[..., None], ke


def check_free_checkpoint(config):
    if 3 not in config.get("modes", []) or config.get("conditioning_version", 1) < CONDITIONING_VERSION:
        raise ValueError("--free-s0 requires a checkpoint trained with corrected mode 3 masks (conditioning_version >= 2)")


def check_rest_checkpoint(config):
    if 4 not in config.get("modes", []) or config.get("conditioning_version", 0) != CONDITIONING_VERSION:
        raise ValueError("Mode 4 now anchors position/yaw only (conditioning_version=4); retrain --modes 2 4. Old stationary-rest checkpoints are incompatible.")

def check_fixed_e_rest_checkpoint(config):
    if 5 not in config.get("modes", []) or config.get("conditioning_version", 1) != CONDITIONING_VERSION:
        raise ValueError("--shared-e-guidance requires mode 5 (known E + rest S0); retrain with --modes ... 4 5")


class Net(nn.Module):
    def __init__(self, H, d, layers):
        super().__init__()
        self.inp = nn.Linear(2 * X_DIM, d)
        self.inp_e = nn.Linear(2 * E_DIM, d)
        self.pos = nn.Parameter(torch.zeros(1, H + 1, d))
        self.temb = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out, self.out_e = nn.Linear(d, X_DIM), nn.Linear(d, E_DIM)
        self.d = d

    def forward(self, x, e, kx, ke, k, pad):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
        t = self.temb(torch.cat([torch.sin(k[:, None] * f), torch.cos(k[:, None] * f)], 1))[:, None]
        h = torch.cat([self.inp_e(torch.cat([e, ke], -1))[:, None], self.inp(torch.cat([x, kx], -1))], 1) + self.pos + t
        h = self.enc(h, src_key_padding_mask=torch.cat([torch.zeros_like(pad[:, :1]), pad], 1))
        return self.out(h[:, 1:]), self.out_e(h[:, 0])


def schedule(n, dev):
    s = torch.arange(n + 1, dtype=torch.float64) / n
    abar = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
    abar = abar / abar[0]
    beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return beta.float().to(dev), torch.cumprod(1 - beta, 0).float().to(dev)


def rebase(x):
    """episode 시작 기준 [B,H,X] -> 각 window 첫 state 기준 (위치·yaw 만 바뀐다)."""
    x = x.clone()
    c0, s0 = x[:, :1, 3], x[:, :1, 4]
    d = x[..., 0:2] - x[:, :1, 0:2]
    c, s = x[..., 3].clone(), x[..., 4].clone()
    x[..., 0], x[..., 1] = c0 * d[..., 0] + s0 * d[..., 1], -s0 * d[..., 0] + c0 * d[..., 1]
    x[..., 3], x[..., 4] = c * c0 + s * s0, s * c0 - c * s0
    return x


def unbase(w, start):
    """window 기준 [B,H,X] -> episode 시작 기준. start = window 첫 state 의 episode 기준 값 [B,X]."""
    w = w.clone()
    c0, s0 = start[:, None, 3], start[:, None, 4]
    px, py, c, s = w[..., 0].clone(), w[..., 1].clone(), w[..., 3].clone(), w[..., 4].clone()
    w[..., 0], w[..., 1] = start[:, None, 0] + c0 * px - s0 * py, start[:, None, 1] + s0 * px + c0 * py
    w[..., 3], w[..., 4] = c * c0 - s * s0, s * c0 + c * s0
    return w
