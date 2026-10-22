"""Minimal ground-geometry feasibility helpers for the 2s TT-to-MJX gate.

This is a bounded, explicit *post-denoising S0 projection*, NOT a
differentiable reward gradient or a correction to predicted TT trajectories.
Do not silently reinterpret projection-assisted MJX as raw TT feasibility.
"""
import numpy as np


def bounded_lift(foot_clearances_m, nonfoot_penetration_m, *,
                 tolerance_m=0.002, margin_m=0.001, max_lift_m=0.04):
    """Return the smallest bounded upward base displacement that clears floor penetration.

    The robot's morphology E, orientation, joints and velocities are preserved.
    Contact/geometry must be recomputed after applying the displacement.
    """
    feet = np.asarray(foot_clearances_m, dtype=float)
    nonfoot = float(nonfoot_penetration_m)
    if (feet.shape != (4,) or not np.isfinite(feet).all()
            or not np.isfinite(nonfoot) or nonfoot < 0
            or not 0 <= tolerance_m < max_lift_m
            or margin_m < 0 or not np.isfinite([tolerance_m, margin_m, max_lift_m]).all()):
        raise ValueError("invalid geometry or vertical projection thresholds")
    penetration = max(0.0, -float(feet.min()), nonfoot)
    needs_change = penetration > tolerance_m
    requested = (penetration + margin_m) if needs_change else 0.0
    allowed = requested <= max_lift_m
    return {
        "needed": bool(needs_change),
        "allowed": bool(allowed),
        "requested_lift_m": float(requested),
        "applied_lift_m": float(requested if allowed else 0.0),
        "reason": ("no_correction_needed" if not needs_change
                   else "within_bounded_lift" if allowed
                   else "required_lift_exceeds_limit"),
        "max_lift_m": float(max_lift_m),
    }


def classify_initial_support(cpu_contacts, mjx_contacts, foot_clearances_m, *,
                             allow_airborne=False, flight_clearance_m=0.005):
    """Allow a physically clear, contact-free start to face real MJX dynamics.

    This does NOT guarantee controlled flight: the subsequent 2s rollout must
    still survive; single-foot/ambiguous contact starts remain rejected.
    """
    c = np.asarray(cpu_contacts, dtype=bool)
    j = np.asarray(mjx_contacts, dtype=bool)
    h = np.asarray(foot_clearances_m, dtype=float)
    if c.shape != (4,) or j.shape != (4,) or h.shape != (4,) or not np.isfinite(h).all():
        raise ValueError("expected four finite foot clearances and four CPU/MJX contact flags")
    grounded = int(c.sum()) >= 2 and int(j.sum()) >= 2
    airborne = (bool(allow_airborne) and not c.any() and not j.any()
                and float(h.min()) >= flight_clearance_m)
    status = "grounded" if grounded else "airborne_unverified" if airborne else "insufficient_support"
    return {"support_ok": bool(grounded or airborne), "initial_support_mode": status,
            "cpu_contacts": int(c.sum()), "mjx_contacts": int(j.sum())}
