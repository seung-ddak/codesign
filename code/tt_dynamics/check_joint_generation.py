"""Gate joint generation before long-horizon reward guidance."""

import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("summary", type=Path)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--min-valid", type=float, default=0.75)
parser.add_argument("--max-path-error", type=float, default=0.25)
args = parser.parse_args()

d = json.loads(args.summary.read_text())
rows = [row for values in d["runs"].values() for row in values]
if not rows:
    raise ValueError("no generated candidates")
# New summaries include all initial physical checks, including the MJX
# foot-ground audit. Older summaries lack this information and fail closed.
raw_valid = [bool(r["initial_physics"].get("physically_valid", False)) for r in rows]
survived = [r["mjx_fell_step"] < 0 for r in rows]
consistent = [r["path_err_m"] <= args.max_path_error for r in rows]
joint_valid = [a and b and c for a, b, c in zip(raw_valid, survived, consistent)]
rates = {"raw_s0_valid": sum(raw_valid) / len(rows), "survived": sum(survived) / len(rows),
         "dynamics_consistent": sum(consistent) / len(rows),
         "joint_valid": sum(joint_valid) / len(rows)}
passed = rates["joint_valid"] >= args.min_valid
out = {"passed": passed, "threshold": args.min_valid, "max_path_error_m": args.max_path_error,
       "candidates": len(rows), "rates": rates, "source": str(args.summary)}
args.out.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
raise SystemExit(0 if passed else 2)
