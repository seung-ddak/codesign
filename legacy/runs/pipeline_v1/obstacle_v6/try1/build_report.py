"""The open report now displays the five-pole experiment; single-pole artifacts stay archived."""
from pathlib import Path
import runpy
root = Path(__file__).resolve().parent
source = root.parent / "obstacle5_v6/try1"
runpy.run_path(str(source / "build_report.py"), run_name="__main__")
report = (source / "report.html").read_text().replace('href="summary.json"', 'href="../obstacle5_v6_try1/summary.json"')
(root / "report.html").write_text(report)
