"""Synthetic CPU smoke checks of actual in-denoising full-reward gradients.

No MuJoCo/JAX/expert files needed. Use real TT + MJX for research results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))

import torch
from joint_diffusion_guidance import joint_diffusion_full_reward


class TinyShapeSensitiveNet(torch.nn.Module):
    def __init__(self, e_index=1):
        super().__init__()
        self.e_index = e_index

    def forward(self, x, e, kx, ke, k, pad):
        # Use active E[1] by default; E[0], E[11], E[12] are inactive
        # coordinates whose morphology reward gradient is intentionally zero.
        ex = torch.ones_like(x) * (0.1 * e[:, self.e_index:self.e_index + 1, None])
        ee = torch.zeros_like(e)
        return ex, ee


class ToyDifferentiableTask:
    def __call__(self, traj, valid):
        # Entire-trajectory differentiable reward, not a single-window reward.
        r = traj[:, :, 2].mean(1) + traj[:, :, 42].mean(1)
        return r, {}


def test_joint_guidance_reaches_noisy_morphology_token():
    n = 1
    a = torch.tensor([0.8, 0.5, 0.2], dtype=torch.float32)
    for scale in (0.0, 1.0):
        x, e, meta = joint_diffusion_full_reward(
            TinyShapeSensitiveNet(), ToyDifferentiableTask(),
            n=n, episode_states=9, horizon=4, state_dim=42,
            action_dim=12, e_dim=13,
            mu=torch.zeros(54), sd=torch.ones(54),
            beta=torch.tensor([0.1, 0.3, 0.5]),
            abar=a, guidance_scale=scale, clip=5.0, seed=17)
        assert x.shape == (1, 9, 54)
        assert e.shape == (1, 13)
        assert torch.isfinite(x).all() and torch.isfinite(e).all()
        assert ((e >= 0.05) & (e <= 0.95)).all()
        assert torch.allclose(e[:, [0, 11, 12]], torch.full((1, 3), 0.5))
        if scale > 0:
            assert meta["e_gradient_nonzero"], meta
            assert len(meta["e_gradient_norm_per_step"]) == 3
        else:
            assert not meta["e_gradient_nonzero"]


def test_joint_guidance_has_deterministic_noise():
    kwargs = dict(n=1, episode_states=7, horizon=4,
                  state_dim=42, action_dim=12, e_dim=13,
                  mu=torch.zeros(54), sd=torch.ones(54),
                  beta=torch.tensor([0.1, 0.3]),
                  abar=torch.tensor([0.9, 0.63]),
                  guidance_scale=0.5, clip=5.0, seed=42)
    a, ea, _ = joint_diffusion_full_reward(TinyShapeSensitiveNet(), ToyDifferentiableTask(), **kwargs)
    b, eb, _ = joint_diffusion_full_reward(TinyShapeSensitiveNet(), ToyDifferentiableTask(), **kwargs)
    assert torch.allclose(a, b)
    assert torch.allclose(ea, eb)


def test_inactive_morphology_dimension_does_not_fake_success():
    _, _, meta = joint_diffusion_full_reward(
        TinyShapeSensitiveNet(e_index=0), ToyDifferentiableTask(),
        n=1, episode_states=9, horizon=4, state_dim=42,
        action_dim=12, e_dim=13,
        mu=torch.zeros(54), sd=torch.ones(54),
        beta=torch.tensor([0.1, 0.3, 0.5]),
        abar=torch.tensor([0.8, 0.5, 0.2]),
        guidance_scale=1.0, clip=5.0, seed=17)
    assert not meta["e_gradient_nonzero"]
    assert all(g == 0.0 for g in meta["e_gradient_norm_per_step"])
