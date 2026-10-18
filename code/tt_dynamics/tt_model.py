"""TT 모델 (diffusion transformer) 과 window 좌표 변환. train_tt.py 와 generate_task.py 가 같이 쓴다."""

import math

import torch
import torch.nn as nn

from rollout_data import E_DIM, X_DIM


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
