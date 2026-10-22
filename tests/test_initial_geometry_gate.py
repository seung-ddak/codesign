"""Regression tests for bounded morphology-specific S0 geometry and flight gating.

These tests intentionally use no MuJoCo/JAX; full MJX gate remains a server test.
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
from initial_geometry_gate import bounded_lift, classify_initial_support


def test_small_penetration_gets_bounded_geometry_derived_lift():
    r = bounded_lift([-0.0139, -0.0019, 0.03, 0.04], 0.008)
    assert r["needed"] and r["allowed"]
    assert abs(r["applied_lift_m"] - 0.0149) < 1e-10
    assert r["reason"] == "within_bounded_lift"


def test_deep_penetration_is_rejected_without_moving_body():
    r = bounded_lift([-0.060, 0.01, 0.02, 0.03], 0.04)
    assert r["needed"] and not r["allowed"]
    assert r["applied_lift_m"] == 0
    assert r["reason"] == "required_lift_exceeds_limit"


def test_clean_geometry_is_never_adjusted():
    r = bounded_lift([-0.001, 0.01, 0.02, 0.03], 0.001)
    assert r["allowed"] and not r["needed"]
    assert r["applied_lift_m"] == 0


def test_flight_is_a_candidate_not_grounded_support():
    feet = [0.018, 0.06, 0.05, 0.05]
    c = [False] * 4
    denied = classify_initial_support(c, c, feet, allow_airborne=False)
    allowed = classify_initial_support(c, c, feet, allow_airborne=True)
    assert not denied["support_ok"]
    assert allowed["support_ok"]
    assert allowed["initial_support_mode"] == "airborne_unverified"


def test_zero_contact_ground_collision_is_not_flight():
    c = [False] * 4
    r = classify_initial_support(c, c, [-0.01, 0.02, 0.04, 0.03], allow_airborne=True)
    assert not r["support_ok"]


def test_one_foot_ambiguous_contact_is_not_accepted_as_flight():
    r = classify_initial_support([True, False, False, False],
                                 [True, False, False, False],
                                 [0, 0.02, 0.03, 0.04], allow_airborne=True)
    assert not r["support_ok"]


def test_two_contacts_still_count_as_grounded():
    r = classify_initial_support([True, True, False, False],
                                 [True, True, False, False],
                                 [0, 0, 0.025, 0.03], allow_airborne=True)
    assert r["support_ok"]
    assert r["initial_support_mode"] == "grounded"


def test_invalid_nonfinite_geometry_is_rejected():
    with pytest.raises(ValueError):
        bounded_lift([np.nan, 0, 0, 0], 0)
