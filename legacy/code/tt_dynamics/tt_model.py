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
    def __init__(self, H, d, layers, film=False):
        super().__init__()
        self.inp = nn.Linear(2 * X_DIM, d)
        self.inp_e = nn.Linear(2 * E_DIM, d)
        self.pos = nn.Parameter(torch.zeros(1, H + 1, d))
        self.temb = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, 8, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out, self.out_e = nn.Linear(d, X_DIM), nn.Linear(d, E_DIM)
        self.d = d
        # film: E token drives a per-layer scale and shift; zero start = no modulation
        self.film = nn.ModuleList(nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2 * d))
                                  for _ in range(layers)) if film else None
        for block in self.film or ():
            nn.init.zeros_(block[-1].weight)
            nn.init.zeros_(block[-1].bias)

    def forward(self, x, e, kx, ke, k, pad):
        half = self.d // 2
        f = torch.exp(-math.log(10000) * torch.arange(half, device=x.device) / half)
        t = self.temb(torch.cat([torch.sin(k[:, None] * f), torch.cos(k[:, None] * f)], 1))[:, None]
        e_token = self.inp_e(torch.cat([e, ke], -1))[:, None]
        h = torch.cat([e_token, self.inp(torch.cat([x, kx], -1))], 1) + self.pos + t
        mask = torch.cat([torch.zeros_like(pad[:, :1]), pad], 1)
        if self.film is None:
            h = self.enc(h, src_key_padding_mask=mask)
        else:
            for layer, block in zip(self.enc.layers, self.film):
                scale, shift = block(e_token + t).chunk(2, -1)
                h = layer(h, src_key_padding_mask=mask) * (1 + scale) + shift
        return self.out(h[:, 1:]), self.out_e(h[:, 0])


def make_net(config):
    return Net(config["horizon"], config["d_model"], config["layers"], film=config.get("film", False))


class EGuidedNet:
    """Strengthen a known E: push the prediction from "E hidden" toward "E given".

    Needs a checkpoint trained with --e-dropout. weight 1 = plain model.
    Rows whose E is being generated (ke = 0) stay unchanged.
    """

    def __init__(self, net, weight):
        self.net, self.weight = net, weight

    def __call__(self, x, e, kx, ke, k, pad):
        ex, ee = self.net(x, e, kx, ke, k, pad)
        known = ke.bool().any(1)
        if self.weight == 1 or not known.any():
            return ex, ee
        hidden, _ = self.net(x, torch.zeros_like(e), kx, torch.zeros_like(ke), k, pad)
        return torch.where(known[:, None, None], hidden + self.weight * (ex - hidden), ex), ee


def schedule(n, dev):
    s = torch.arange(n + 1, dtype=torch.float64) / n
    abar = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
    abar = abar / abar[0]
    beta = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
    return beta.float().to(dev), torch.cumprod(1 - beta, 0).float().to(dev)


def unit_yaw(c, s, eps=1e-6):
    """Use only SO(2) rotation when stitching TT windows.

    Predicted cos/sin channels are unconstrained by diffusion. Fall back to
    identity for near-zero headings, rather than scaling travelled distances.
    This helper preserves autograd for non-degenerate predictions.
    """
    norm = torch.sqrt(c.square() + s.square()).clamp_min(eps)
    good = (c.square() + s.square()) > eps * eps
    return torch.where(good, c / norm, torch.ones_like(c)), torch.where(
        good, s / norm, torch.zeros_like(s))


def rebase(x):
    """Episode frame -> first-state frame; yaw must be a UNIT rotation."""
    x = x.clone()
    c0, s0 = unit_yaw(x[:, :1, 3], x[:, :1, 4])
    d = x[..., 0:2] - x[:, :1, 0:2]
    c, s = unit_yaw(x[..., 3].clone(), x[..., 4].clone())
    x[..., 0], x[..., 1] = c0 * d[..., 0] + s0 * d[..., 1], -s0 * d[..., 0] + c0 * d[..., 1]
    x[..., 3], x[..., 4] = c * c0 + s * s0, s * c0 - c * s0
    return x


def unbase(w, start):
    """Window frame -> episode frame; never scale positions via yaw magnitude."""
    w = w.clone()
    c0, s0 = unit_yaw(start[:, None, 3], start[:, None, 4])
    px, py = w[..., 0].clone(), w[..., 1].clone()
    c, s = unit_yaw(w[..., 3].clone(), w[..., 4].clone())
    w[..., 0], w[..., 1] = start[:, None, 0] + c0 * px - s0 * py, start[:, None, 1] + s0 * px + c0 * py
    w[..., 3], w[..., 4] = c * c0 - s * s0, s * c0 + c * s0
    return w
