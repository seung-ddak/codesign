"""Evaluate existing FAST joint generation + MJX artifacts, without rerunning them.

Use after generate_task.py has written summary.json and scale_<value>.npz.
A failed candidate is NEVER counted as a zero-reward successful MJX rollout.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def evaluate(root: Path, guided_scale: float, seconds: float):
    summary = json.loads((root / "summary.json").read_text())
    tag = f"scale_{guided_scale:g}"
    base_tag = "scale_0"
    if guided_scale <= 0:
        raise ValueError("guided_scale must be positive for a guidance comparison")
    if tag not in summary["runs"]:
        raise ValueError(f"missing guided run {tag}")
    if base_tag not in summary["runs"]:
        raise ValueError("missing scale_0 control with the same noise seed")

    gr = summary["runs"][tag]
    audit = summary.get("joint_full_guidance_audit", {}).get(tag, {})
    with np.load(root / f"{tag}.npz") as z:
        e = z["E"].copy()
        pred = z["pred"].copy()
    with np.load(root / f"{base_tag}.npz") as z:
        eb = z["E"].copy()
        predb = z["pred"].copy()

    expected_steps = round(seconds / 0.02)
    dimensions_ok = (e.ndim == 2 and e.shape[1] == 13
                     and pred.ndim == 3 and pred.shape[2] == 54
                     and pred.shape[0] == e.shape[0] == len(gr)
                     and pred.shape[1] == expected_steps
                     and e.shape == eb.shape and pred.shape == predb.shape)
    finite = (dimensions_ok and np.isfinite(e).all() and np.isfinite(pred).all()
              and np.isfinite(eb).all() and np.isfinite(predb).all())
    changes = {
        "E_l2": float(np.linalg.norm(e - eb)) if finite else None,
        "S_l2": float(np.linalg.norm(pred[..., :42] - predb[..., :42])) if finite else None,
        "A_l2": float(np.linalg.norm(pred[..., 42:] - predb[..., 42:])) if finite else None,
    }
    no_fall_valid = [
        (r.get("physically_valid_s0", False)
         and not r.get("execution_skipped", False)
         and r.get("mjx_fell_step") == -1
         and float(r.get("mjx_valid_s", 0)) >= seconds - 0.010
         and r.get("mjx_total") is not None
         and math.isfinite(r["mjx_total"]))
        for r in gr
    ]

    components = {}
    for component in ("e", "s", "a"):
        norms = audit.get(f"{component}_gradient_norm_per_step", [])
        components[component] = (
            audit.get(f"{component}_gradient_nonzero") is True
            and len(norms) == summary.get("joint_full_guidance_audit", {}).get(tag, {}).get("diffusion_steps")
            and all(math.isfinite(v) and v >= 0 for v in norms)
            and any(v > 1e-12 for v in norms)
        )

    raw_completed = [
        ok and not row.get("initial_physics", {}).get("geometry_adjusted", False)
        for ok, row in zip(no_fall_valid, gr)
    ]
    projected_completed = [
        ok and row.get("initial_physics", {}).get("geometry_adjusted", False)
        for ok, row in zip(no_fall_valid, gr)
    ]
    checks = {
        "real_checkpoint_joint_generation_finite": bool(finite),
        "E_reward_gradient_nonzero": bool(components["e"]),
        "S_reward_gradient_nonzero": bool(components["s"]),
        "A_reward_gradient_nonzero": bool(components["a"]),
        "guidance_changes_generated_E": bool(finite and changes["E_l2"] > 1e-8),
        "guidance_changes_generated_S": bool(finite and changes["S_l2"] > 1e-8),
        "guidance_changes_generated_A": bool(finite and changes["A_l2"] > 1e-8),
        "at_least_one_valid_initial_state": any(r.get("physically_valid_s0", False) for r in gr),
        "at_least_one_complete_MJX_open_loop": any(no_fall_valid),
    }
    return {
        "stage": "FAST joint E/S/A generation plus MJX open-loop",
        "task": summary.get("task"),
        "seconds": seconds,
        "guided_scale": guided_scale,
        "checkpoint": summary.get("ckpt"),
        "checks": checks,
        "all_pass": all(checks.values()),
        "guidance_change_l2": changes,
        "guided_candidates": len(gr),
        "guided_valid_s0_count": sum(bool(r.get("physically_valid_s0")) for r in gr),
        "guided_completed_MJX_count": sum(no_fall_valid),
        "guided_completed_unmodified_MJX_count": sum(raw_completed),
        "guided_completed_projection_assisted_MJX_count": sum(projected_completed),
        "guided_geometry_adjusted_count": sum(
            bool(r.get("initial_physics", {}).get("geometry_adjusted", False)) for r in gr),
        "guided_airborne_start_count": sum(
            r.get("initial_physics", {}).get("initial_support_mode") == "airborne_unverified"
            for r in gr),
        "strict_raw_TT_to_MJX_verified": bool(all(checks.values()) and any(raw_completed)),
        "guided_rejected_count": sum(bool(r.get("execution_skipped")) for r in gr),
        "guided_rewards": [r.get("mjx_total") for r in gr],
        "baseline_rewards": [r.get("mjx_total") for r in summary["runs"][base_tag]],
        "note": ("The gate may pass with post-denoising geometry projection, in which case "
                 "only projection-assisted MJX execution was verified, NOT raw TT physical "
                 "generation. A zero-contact airborne start is a trial, not proof of stable "
                 "flight. This cannot establish task-optimality or 10s success."),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--guided-scale", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()
    report = evaluate(args.out, args.guided_scale, args.seconds)
    (args.out / "smoke_check.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    for check, passed in report["checks"].items():
        print(f"{'PASS' if passed else 'FAIL'} {check}")
    print(f"E/S/A change L2: {report['guidance_change_l2']}")
    print(f"Unmodified 2s completions: {report['guided_completed_unmodified_MJX_count']}; "
          f"projection-assisted completions: {report['guided_completed_projection_assisted_MJX_count']}; "
          f"airborne candidates: {report['guided_airborne_start_count']}")
    print(f"Valid S0: {report['guided_valid_s0_count']}/{report['guided_candidates']}; "
          f"MJX completed: {report['guided_completed_MJX_count']}/{report['guided_candidates']}")
    print(f"Detailed report: {args.out / 'smoke_check.json'}")
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
