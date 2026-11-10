"""Mode-4 initial S0 conditioning: no rare forced-rest state.

Pure-Torch contract. This does not prove generated S0 is physically safe;
that is assessed by CPU MuJoCo audit and MJX execution.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))

import pytest
import torch

from tt_model import (CONDITIONING_VERSION, ANCHOR_KNOWN, anchor_s0,
                      check_rest_checkpoint, condition_masks, rebase)
from rollout_data import E_DIM, S_DIM, X_DIM


def test_anchor_has_only_heading_frame_coordinates():
    assert CONDITIONING_VERSION == 4
    assert torch.equal(ANCHOR_KNOWN.nonzero().flatten(), torch.tensor([0, 1, 3, 4]))
    expected = torch.zeros(S_DIM)
    expected[3] = 1.0
    assert torch.equal(anchor_s0(), expected)


def test_mode4_learns_dynamic_start_features_from_data():
    valid = torch.ones(2, 5, dtype=torch.bool)
    mode = torch.tensor([4, 2])
    kx, ke = condition_masks(mode, valid)
    assert torch.equal(kx[0, 0, :S_DIM], ANCHOR_KNOWN)
    assert torch.equal(kx[1, 0, :S_DIM], torch.ones(S_DIM))
    assert torch.equal(ke[0], torch.zeros(E_DIM))
    assert torch.equal(ke[1], torch.ones(E_DIM))
    assert kx[0, 0, 2].item() == 0.0  # height
    assert kx[0, 0, 5:8].sum().item() == 0.0  # body tilt / gravity
    assert kx[0, 0, 8:14].sum().item() == 0.0  # velocities
    assert kx[0, 0, 14:38].sum().item() == 0.0  # joints and joint velocities
    assert kx[0, 0, 38:42].sum().item() == 0.0  # contact
    assert kx[0, :, S_DIM:].sum().item() == 0.0  # action is generated


def test_random_heading_windows_always_match_anchor():
    x = torch.zeros(3, 5, X_DIM)
    x[:, :, 0] = torch.arange(5).float()
    x[:, :, 1] = torch.tensor([0.1, -0.2, 0.3])[None].T
    yaw = torch.tensor([0.3, -1.0, 2.2])
    x[:, :, 3] = yaw.cos()[:, None]
    x[:, :, 4] = yaw.sin()[:, None]
    x[:, :, 8] = 0.4  # real moving s0 must stay unknown
    rebased = rebase(x)
    assert torch.allclose(rebased[:, 0, :2], torch.zeros(3, 2), atol=1e-5)
    assert torch.allclose(rebased[:, 0, 3], torch.ones(3), atol=1e-5)
    assert torch.allclose(rebased[:, 0, 4], torch.zeros(3), atol=1e-5)


def test_old_stationary_mode4_checkpoint_must_be_retrained():
    with pytest.raises(ValueError, match="conditioning_version=4"):
        check_rest_checkpoint({"modes": [2, 4], "conditioning_version": 3})
    check_rest_checkpoint({"modes": [2, 4], "conditioning_version": 4})
