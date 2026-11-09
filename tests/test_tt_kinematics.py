"""Regression tests for physical consistency of TT denoised states."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
from tt_kinematics import denoised_kinematic_terms, kinematic_terms


def physical_motion(yaw=0.0):
    x = torch.zeros(2, 6, 54)
    x[..., 3] = torch.cos(torch.tensor(yaw))
    x[..., 4] = torch.sin(torch.tensor(yaw))
    x[..., 8] = 0.5
    x[..., 26:38] = 1.0
    for t in range(6):
        x[:, t, 0] = 0.5 * t * 0.02 * torch.cos(torch.tensor(yaw))
        x[:, t, 1] = 0.5 * t * 0.02 * torch.sin(torch.tensor(yaw))
        x[:, t, 14:26] = t * 0.02
    return x


@pytest.mark.parametrize("yaw", [0.0, 0.7, -1.5])
def test_constant_velocity_motion_exact_consistency(yaw):
    x = physical_motion(yaw)
    out = kinematic_terms(x, torch.ones(2, 6, dtype=torch.bool))
    assert out["xy_rmse_cm"].item() < 1e-5
    assert out["joint_rmse_deg"].item() < 1e-5


def test_invalid_transition_not_counted_and_gradients_flow():
    x = physical_motion().requires_grad_()
    xmod = x.clone()
    xmod[:, 2, 0] += .1
    xmod[:, 3, 14] += .2
    valid = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    out = kinematic_terms(xmod, valid)
    assert out["pair_count"].item() == 6
    assert out["xy_loss"].item() > 0
    assert out["joint_loss"].item() > 0
    (out["xy_loss"] + out["joint_loss"]).backward()
    assert x.grad.abs().sum().item() > 0
    assert torch.isfinite(x.grad).all()


def test_padding_mask_drops_both_endpoints():
    x = physical_motion()
    good = torch.tensor([[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    x[:, 2:] = 1000
    out = kinematic_terms(x, good)
    assert out["pair_count"].item() == 1
    assert out["xy_rmse_cm"].item() < 1e-5
    assert out["joint_rmse_deg"].item() < 1e-5
    zero = kinematic_terms(x, torch.zeros_like(good))
    assert zero["pair_count"].item() == 0
    assert zero["xy_loss"].item() == 0


def test_recovered_x0_conditioned_and_zero_weight_regression():
    x0 = physical_motion()
    mu, sd = torch.zeros(54), torch.ones(54)
    known = torch.zeros_like(x0)
    known[:, 0, :42] = 1
    valid = torch.ones(2, 6, dtype=torch.bool)
    ab = torch.tensor([0.6, 0.8])
    noise = torch.randn_like(x0)
    xt = torch.sqrt(ab[:, None, None])*x0 + torch.sqrt(1-ab[:, None, None])*noise
    epsilon = noise.detach().clone().requires_grad_()
    kin = denoised_kinematic_terms(xt, epsilon, x0, known, valid, ab, mu, sd, 100)
    assert kin["xy_rmse_cm"].item() < 1e-4
    assert kin["joint_rmse_deg"].item() < 1e-4
    base_loss = (epsilon ** 2).mean()
    total = base_loss + 0.0*kin["xy_loss"] + 0.0*kin["joint_loss"]
    assert torch.equal(base_loss, total)


def test_error_from_unconditioned_model_propagates_gradient():
    x0 = physical_motion()
    eps = torch.randn_like(x0, requires_grad=True)
    ab = torch.tensor([0.6, 0.6])
    xt = torch.sqrt(ab[:, None, None]) * x0
    kin = denoised_kinematic_terms(
        xt, eps, x0, torch.zeros_like(x0),
        torch.ones(2,6,dtype=torch.bool), ab,
        torch.zeros(54), torch.ones(54), 10.)
    (kin["xy_loss"]+kin["joint_loss"]).backward()
    assert torch.isfinite(eps.grad).all()
    assert eps.grad.abs().sum().item() > 0

def test_min_snr_damps_high_noise_penalty_without_hiding_raw_rmse():
    x0 = physical_motion()
    eps = torch.randn_like(x0)
    xt = x0.clone()
    mask = torch.ones(2, 6, dtype=torch.bool)
    out = denoised_kinematic_terms(
        xt, eps, x0, torch.zeros_like(x0), mask,
        torch.tensor([0.0001, 0.0001]), torch.zeros(54), torch.ones(54), 1.)
    assert out["xy_loss"].item() < 50
    assert out["joint_loss"].item() < 50
    assert out["xy_rmse_cm"].item() >= 0
