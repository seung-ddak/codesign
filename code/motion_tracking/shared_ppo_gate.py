"""Decide whether to manually unlock the next morphology curriculum stage.

No training is started or modified by this gate. Run it on full 10-second MJX
metrics, then resume a *saved shared-PPO checkpoint* with an explicit unlock
mask only after exit code 0. Exit code 2 means stop and diagnose.
"""
import argparse
import json
import math
from pathlib import Path


def assess(rows, designs, min_trials=12, min_survival=0.8,
           max_body_rmse_m=0.20, max_joint_rmse_deg=12.0,
           max_orientation_deg=10.0):
    results = {}
    for design in designs:
        subset = [r for r in rows if r.get("design") == design and r.get("train_motion") is True]
        passed = [r for r in subset if r.get("survived_10s") is True]
        notes = []
        if len(subset) < min_trials:
            notes.append(f"Only {len(subset)} trials; need {min_trials}")
        rate = len(passed) / len(subset) if subset else 0.0
        if rate < min_survival:
            notes.append(f"Survival {rate:.3f} < {min_survival:.3f}")
        measures = {
            "body_position_rmse_m": max_body_rmse_m,
            "joint_rmse_deg": max_joint_rmse_deg,
            "orientation_mean_deg": max_orientation_deg,
        }
        means = {}
        for name, limit in measures.items():
            values = [r.get(name) for r in passed]
            if not values or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                means[name] = None
                notes.append(f"Missing/invalid survivor measurement: {name}")
            else:
                mean = sum(values) / len(values)
                means[name] = mean
                if mean > limit:
                    notes.append(f"{name} {mean:.4f} > {limit:.4f}")
        results[design] = {
            "n": len(subset), "survived": len(passed),
            "survival_rate": rate, "survivor_means": means,
            "passed": not notes, "issues": notes,
        }
    return {"passed": bool(designs) and all(v["passed"] for v in results.values()),
            "designs": results}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--designs", nargs="+", required=True,
                    help="ALL already-unlocked training designs; no held-out designs")
    ap.add_argument("--min-trials", type=int, default=12)
    ap.add_argument("--min-survival", type=float, default=0.8)
    ap.add_argument("--max-body-rmse-m", type=float, default=0.20)
    ap.add_argument("--max-joint-rmse-deg", type=float, default=12.0)
    ap.add_argument("--max-orientation-deg", type=float, default=10.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.min_trials < 1 or not 0 < args.min_survival <= 1:
        ap.error("Gate requires min_trials>=1 and 0<min_survival<=1")
    if len(set(args.designs)) != len(args.designs):
        ap.error("Duplicate design names")
    rows = json.loads(args.metrics.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        ap.error("Expected a list of evaluation records")
    outcome = assess(rows, args.designs, args.min_trials, args.min_survival,
                     args.max_body_rmse_m, args.max_joint_rmse_deg,
                     args.max_orientation_deg)
    print(json.dumps(outcome, ensure_ascii=False, indent=2, allow_nan=False))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    if not outcome["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
