"""Choose a wide-design expert checkpoint by held-out evaluation on training morphologies."""

import csv
import json
import subprocess
import sys
from pathlib import Path


run = Path(sys.argv[1])
checkpoints = sorted((p for p in (run / "files/checkpoints").iterdir()
                      if p.name.isdigit() and int(p.name) > 0 and (p / "_CHECKPOINT_METADATA").is_file()),
                     key=lambda p: int(p.name))
if not checkpoints:
    raise SystemExit("No completed nonzero checkpoint")

# e003 is held out; it must not influence checkpoint selection.
designs = ("e000", "e001", "e002", "e008", "e012", "e016")
records = []
for checkpoint in checkpoints:
    out = run / "checkpoint_eval" / checkpoint.name
    rows = []
    for design in designs:
        csv_path = out / design / "motion_eval.csv"
        if not csv_path.is_file():
            subprocess.run([sys.executable, "code/motion_tracking/eval_motions.py", str(run),
                            "--design", design, "--checkpoint", str(checkpoint),
                            "--output", str(out), "--no-render"], check=True)
        with csv_path.open(newline="") as file:
            rows.extend(csv.DictReader(file))
    if len(rows) != len(designs):
        raise ValueError(f"Expected one motion per design at {checkpoint}")
    passed = sum(int(row["pass_all"]) for row in rows)
    survived = sum(int(row["survived"]) for row in rows)
    error = sum(float(row["joint_rms_deg"]) / 5 + float(row["foot_err_cm"]) / 3 +
                float(row["orn_err_deg"]) / 5 for row in rows)
    records.append({"step": int(checkpoint.name), "pass_count": passed,
                    "survived_count": survived, "normalized_error": error})
    print(f"{checkpoint.name}: pass={passed}/{len(designs)} survived={survived}/{len(designs)} error={error:.2f}", file=sys.stderr)

best = max(records, key=lambda r: (r["pass_count"], r["survived_count"], -r["normalized_error"]))
(run / "checkpoint_selection.json").write_text(json.dumps({"screen_designs": designs,
    "selected_step": best["step"], "checkpoints": records}, indent=2) + "\n")
print(run / "files/checkpoints" / str(best["step"]))
