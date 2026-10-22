"""Fail-closed gate for short joint generation; null MJX histories are expected failures."""
import argparse
import json
import math
from pathlib import Path


def evaluate(summary, *, min_valid=0.75, max_path_error=0.25):
    if not 0 <= min_valid <= 1 or max_path_error < 0:
        raise ValueError("invalid gate thresholds")
    rows = [r for values in summary.get("runs", {}).values() for r in values]
    if not rows:
        return {"passed": False, "reason": "no generated candidates", "candidates": 0, "rates": {}}

    def physical_ok(r):
        info = r.get("initial_physics")
        return isinstance(info, dict) and info.get("physically_valid") is True and r.get("execution_skipped") is not True

    def survived_ok(r):
        if r.get("execution_skipped") is True:
            return False
        t = r.get("mjx_fell_step")
        return isinstance(t, (int, float)) and math.isfinite(t) and t == -1

    def path_ok(r):
        t = r.get("path_err_m")
        return isinstance(t, (int, float)) and math.isfinite(t) and 0 <= t <= max_path_error

    raw = [physical_ok(r) for r in rows]
    survived = [survived_ok(r) for r in rows]
    consistent = [path_ok(r) for r in rows]
    valid = [all(x) for x in zip(raw, survived, consistent)]
    rates = {
        "initial_physics_valid": sum(raw) / len(rows),
        "survived": sum(survived) / len(rows),
        "dynamics_consistent": sum(consistent) / len(rows),
        "joint_valid": sum(valid) / len(rows),
    }
    return {
        "passed": bool(rates["joint_valid"] >= min_valid),
        "reason": "gate checked; skipped/null/nonfinite results count as failures",
        "threshold": min_valid, "max_path_error_m": max_path_error,
        "candidates": len(rows), "rejected_or_missing_execution": sum(
            r.get("execution_skipped", False) or r.get("mjx_fell_step") is None for r in rows),
        "rates": rates,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("summary", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-valid", type=float, default=0.75)
    p.add_argument("--max-path-error", type=float, default=0.25)
    args = p.parse_args()
    d = json.loads(args.summary.read_text())
    out = evaluate(d, min_valid=args.min_valid, max_path_error=args.max_path_error)
    out["source"] = str(args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))
    return 0 if out["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
