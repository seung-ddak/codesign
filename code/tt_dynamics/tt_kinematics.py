"""Differentiable short-step consistency checks for the 50-Hz TT state token.

S token layout: xy(0:2), z(2), relative yaw cos/sin(3:5), gravity(5:8),
heading-frame linear velocity(8:11), body angular velocity(11:14),
joint angles(14:26), joint velocities(26:38), contacts(38:42).
All losses operate on *physical* denormalized predictions, never on epsilon.
"""

import math

import torch

DT = 0.02
XY_REFERENCE_M = 0.02   # normalize an XY per-axis displacement residual by 2 cm
JOINT_REFERENCE_RAD = 0.1  # normalize joint-angle residual by 0.1 rad


def kinematic_terms(x, valid, *, dt=DT):
    """Return differentiable normalized losses and unscaled residual statistics.

    x: [B,T,54] physical state+action, in episode/window-aligned XY coordinates.
    valid: [B,T] bool mask; a transition counts only if both endpoints are valid.
    The yaw channel encodes heading relative to the episode/window start.
    """
    if x.ndim != 3 or x.shape[-1] != 54 or valid.shape != x.shape[:2]:
        raise ValueError("expected x[B,T,54] and valid[B,T]")
    if x.shape[1] < 2 or dt <= 0:
        raise ValueError("T must be >=2 and dt must be positive")

    yaw = torch.atan2(x[..., 4], x[..., 3])
    c, s = torch.cos(yaw), torch.sin(yaw)
    vh = x[..., 8:10]
    v = torch.stack((c * vh[..., 0] - s * vh[..., 1],
                     s * vh[..., 0] + c * vh[..., 1]), dim=-1)
    dp = x[:, 1:, 0:2] - x[:, :-1, 0:2]
    expected_dp = (v[:, 1:] + v[:, :-1]) * (0.5 * dt)
    xy_residual = dp - expected_dp

    dq = x[:, 1:, 14:26] - x[:, :-1, 14:26]
    expected_dq = (x[:, 1:, 26:38] + x[:, :-1, 26:38]) * (0.5 * dt)
    joint_residual = dq - expected_dq

    pair_mask = (valid[:, 1:] & valid[:, :-1]).to(x.dtype)
    count = pair_mask.sum()
    xy_sq_sum = ((xy_residual ** 2).sum(-1) * pair_mask).sum()
    joint_sq_sum = ((joint_residual ** 2).sum(-1) * pair_mask).sum()
    # count==0: all outputs are zero, and derivatives are well-defined.
    denom = count.clamp(min=1)
    xy_axis_mse = xy_sq_sum / (2 * denom)
    joint_axis_mse = joint_sq_sum / (12 * denom)
    return {
        "xy_loss": xy_axis_mse / (XY_REFERENCE_M ** 2),
        "joint_loss": joint_axis_mse / (JOINT_REFERENCE_RAD ** 2),
        "xy_rmse_cm": torch.sqrt(xy_sq_sum / denom) * 100.0,
        "joint_rmse_deg": torch.sqrt(joint_axis_mse) * (180.0 / math.pi),
        "xy_sq_sum": xy_sq_sum,
        "joint_sq_sum": joint_sq_sum,
        "pair_count": count,
    }


def denoised_kinematic_terms(xt, predicted_eps, clean_x, known_mask,
                              valid, alpha_bar, mu, sd, clip, *, dt=DT):
    """Compute physical consistency of the reconstructed conditional x0.

    The reconstruction is clamped exactly as at inference. Known (inpainted)
    coordinates are replaced with clean_x so they are not penalized as free
    predictions. Returns metrics; caller decides the loss weights.
    """
    ab = alpha_bar.reshape(-1, 1, 1)
    xhat = ((xt - torch.sqrt(1 - ab) * predicted_eps) / torch.sqrt(ab)).clamp(-clip, clip)
    xhat = torch.where(known_mask > 0, clean_x, xhat)
    physical = xhat * sd.view(1, 1, -1) + mu.view(1, 1, -1)
    return kinematic_terms(physical, valid, dt=dt)
