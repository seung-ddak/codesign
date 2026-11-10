"""CPU regression tests for physical orientation and nonnegative guidance reward."""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
from tt_model import unbase, rebase, unit_yaw
from task_reward import TrackReward


def test_window_transform_preserves_distance_with_nonunit_heading():
    w = torch.zeros(1, 2, 54)
    w[:, :, 0] = torch.tensor([0., 1.])
    w[:, :, 3] = 1
    start = torch.zeros(1, 54)
    start[0, 3] = 0.7
    start[0, 0] = 2
    out = unbase(w, start)
    assert torch.allclose(out[0, :, 0], torch.tensor([2., 3.]))
    assert torch.allclose(out[0, :, 3], torch.ones(2))


def test_rebase_unbase_consistent_with_noisy_heading():
    x = torch.zeros(1, 3, 54)
    x[0, :, 0] = torch.tensor([10., 10., 9.])
    x[0, :, 1] = torch.tensor([2., 3., 3.])
    x[0, :, 3] = 0.
    x[0, :, 4] = -1.2
    w = rebase(x)
    restored = unbase(w, x[:, 0])
    assert torch.allclose(restored[0, :, 0:2], x[0, :, 0:2], atol=1e-6)
    assert torch.allclose(torch.linalg.vector_norm(restored[..., 3:5], dim=-1), torch.ones(1, 3))


def test_near_zero_yaw_safe_identity_and_gradients():
    c, s = unit_yaw(torch.tensor([0., 0.7]), torch.tensor([0., 0.]))
    assert torch.allclose(c, torch.ones_like(c))
    assert torch.allclose(s, torch.zeros_like(s))
    x = torch.tensor([0.5], requires_grad=True)
    cs, sn = unit_yaw(x, x)
    (cs + sn).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_invalid_gravity_cannot_create_negative_fall_penalty():
    r = TrackReward()
    s = torch.zeros(1, 3, 54)
    s[..., 2] = 0.3
    s[..., 7] = -2.0
    mask = torch.ones(1, 3, dtype=torch.bool)
    penalty = r.p_fall(s, mask)
    assert penalty.item() >= 0
    assert r(s, mask)[0].item() <= 0
    s[..., 7] = -1.0
    assert abs(r.p_fall(s, mask).item()) < 1e-7
    s[..., 7] = 0.0
    assert r.p_fall(s, mask).item() > 0
