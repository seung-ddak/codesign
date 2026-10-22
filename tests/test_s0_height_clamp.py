"""Small synthetic CPU regression tests for the bounded S0 clamp gate."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
from audit_s0_height_clamp import inspect_s0_height


def _batch(height):
    x = np.zeros((2, 100, 54), dtype=np.float32)
    x[:, 0, 2] = height
    return x


def test_identical_heights_can_be_lower_clamp():
    r = inspect_s0_height(
        np.array([0., 0., 0.25], dtype=np.float32),
        np.array([1., 1., 0.10], dtype=np.float32),
        1.,
        {"scale_0": _batch(0.15), "scale_1": _batch(0.15)})
    assert r["all_initial_heights_at_lower_clip"]
    assert r["any_lower_clip_hit"]
    assert abs(r["physical_lower_clip_m"] - 0.15) < 1e-7


def test_identical_heights_do_not_always_mean_clamp():
    r = inspect_s0_height(
        np.array([0., 0., 0.25], dtype=np.float32),
        np.array([1., 1., 0.10], dtype=np.float32),
        1.,
        {"scale_0": _batch(0.20), "scale_1": _batch(0.20)})
    assert not r["any_lower_clip_hit"]
    assert not r["all_initial_heights_at_lower_clip"]


def test_invalid_normalization_must_fail():
    with pytest.raises(ValueError):
        inspect_s0_height(np.zeros(54), np.zeros(54), 1.,
                          {"scale_0": _batch(0.2)})
