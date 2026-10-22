"""TT 모델 (diffusion transformer) 과 window 좌표 변환. train_tt.py 와 generate_task.py 가 같이 쓴다."""

import math

import torch
import torch.nn as nn

from rollout_data import E_DIM, S_DIM, X_DIM


CONDITIONING_VERSION = 3


# mode 4 의 s0 에서 아는 칸 = 형상과 무관한 물리 사실 (위치·yaw 원점, 중력, 선·각속도, 관절속도, 접지). 높이 2 와 관절각 14:26 은 E 와 함께 생성
REST_KNOWN = torch.ones(S_DIM)
REST_KNOWN[2] = 0
REST_KNOWN[14:26] = 0


def rest_s0():
    """정지 출발 s0 의 아는 칸 값 [S_DIM]: 원점, yaw 0, 몸통 수직 (중력 0,0,-1), 속도 0, 발 4 개 접지. 모르는 칸은 0."""
    s = torch.zeros(S_DIM)
    s[3], s[7], s[38:42] = 1, -1, 1
    return s


def condition_masks(mode, valid):
    """Known-value masks: 0=s0, 1=E+s0+A, 2=E+s0, 3=free, 4=rest (generate E), 5=E+rest (fix E, generate s0/S/A)."""
    n, horizon = valid.shape
    kx = torch.zeros(n, horizon, X_DIM, device=valid.device)
    s0_known = torch.where(((mode == 4) | (mode == 5))[:, None], REST_KNOWN.to(valid.device), torch.ones(S_DIM, device=valid.device))
    kx[:, 0, :S_DIM] = (mode != 3).float()[:, None] * s0_known
    kx[:, :, S_DIM:] = (mode == 1).float()[:, None, None]
    ke = ((mode == 1) | (mode == 2) | (mode == 5)).float()[:, None].expand(n, E_DIM)
    return kx * valid[..., None], ke


def check_free_checkpoint(config):
    if 3 not in config.get("modes", []) or config.get("conditioning_version", 1) < CONDITIONING_VERSION:
        raise ValueError("--free-s0 requires a checkpoint trained with corrected mode 3 masks (conditioning_version >= 2)")


def check_rest_checkpoint(config):
    if 4 not in config.get("modes", []):
        raise ValueError("--rest-s0 requires a checkpoint trained with mode 4 (--modes ... 4)")

def check_fixed_e_rest_checkpoint(config):
    if 5 not in config.get("modes", []) or config.get("conditioning_version", 1) < 3:
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
