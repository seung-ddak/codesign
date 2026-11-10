"""수집 점검 data_audit.json 생성 + conditions.json 의 데이터 sha256 채움 (v10 data_audit.json 과 같은 필드)."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
conditions = json.loads((RUN / "conditions.json").read_text())
args = conditions["training_args"]
files = [Path(a) for a in args[args.index("--data") + 1:] if a.endswith(".npz")]

rows, E = [], []
for f in files:
    z = np.load(ROOT / f, allow_pickle=True)
    valid = z["valid"].astype(bool)
    lengths = valid.sum(1)
    rows.append(dict(design=str(z["design"]), episodes=int(len(valid)), motions=int(len(z["motions"])),
                     valid_controls=int(valid.sum()), falls=int(z["fell"].sum()), minimum_length=int(lengths.min()),
                     median_length=float(np.median(lengths)), group_ids_unique=int(len(np.unique(z["group_id"]))),
                     dt=float(z["dt"]), kp=float(z["kp"]) if "kp" in z.files else None))
    E.append(np.asarray(z["E"], np.float64))
E = np.stack(E)
varying = int((E.std(0) > 1e-9).sum())
rank = int(np.linalg.matrix_rank(E - E.mean(0), tol=1e-6))


def source_stats(rel):
    paths = sorted((ROOT / rel).glob("e*/primitive_seed0.npz"))
    if not paths:
        return None
    ep = ctrl = 0
    for p in paths:
        v = np.load(p, allow_pickle=True)["valid"].astype(bool)
        ep += len(v)
        ctrl += int(v.sum())
    return dict(files=len(paths), episodes=ep, valid_controls=ctrl)


audit = dict(files=len(rows), episodes=sum(r["episodes"] for r in rows), valid_controls=sum(r["valid_controls"] for r in rows),
             falls=sum(r["falls"] for r in rows), rows=rows, morphology_parameter_rank=rank, varying_parameter_count=varying,
             baseline_sources={k: v for k in ("wide16_action/rollouts", "e_swap/full", "pd_rollouts")
                               if (v := source_stats("runs/pipeline_v1/" + k)) is not None})
(RUN / "data_audit.json").write_text(json.dumps(audit, indent=2))
conditions["sha256"] = {str(f): hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in files}
(RUN / "conditions.json").write_text(json.dumps(conditions, indent=2, ensure_ascii=False))
assert audit["files"] == conditions["expected_data_files"], audit["files"]
print(json.dumps({k: v for k, v in audit.items() if k != "rows"}, ensure_ascii=False))
