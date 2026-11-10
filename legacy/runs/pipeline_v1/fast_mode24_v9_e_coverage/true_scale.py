"""How big is the true moving E effect (same A, MJX) compared with TT's own moving error? CPU only."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "code/tt_dynamics")
import e_sensitivity as es
from tt_model import rebase
R = Path("runs/pipeline_v1/fast_mode24_v9_e_coverage")
ck = torch.load(R / "tt_joint/checkpoint.pt", map_location="cpu", weights_only=False)
mu, sd = [torch.as_tensor(np.asarray(v.cpu() if hasattr(v, "cpu") else v), dtype=torch.float32) for v in (ck["mu"], ck["sd"])]
eps = [{k: z[k] for k in ("qpos", "qvel", "contact", "target_q", "valid")} for z in map(np.load, sorted((R / "data_test").glob("n*/primitive_seed0.npz")))]
rng = np.random.default_rng(0); a, b = [], []
for row in rng.choice(len(eps[0]["valid"]), 200, replace=False):
    alive = [i for i in range(len(eps)) if eps[i]["valid"][row].sum() >= 62]
    if len(alive) > 1:
        i, j = rng.choice(alive, 2, replace=False)
        a.append(es.window(eps[i], row, 30, 32)); b.append(es.window(eps[j], row, 30, 32))
xa, xb = rebase(torch.tensor(np.stack(a), dtype=torch.float32)), rebase(torch.tensor(np.stack(b), dtype=torch.float32))
mov = lambda x: x - x.mean(1, keepdim=True)
out = {"pairs": len(a)}
for g, (lo, hi) in es.GROUPS.items():
    if g in ("angvel", "jointvel", "contact"): continue
    s = sd[lo:hi]; d = (xb - xa)[:, 1:, lo:hi] / s; one = mov(xa[:, 1:, lo:hi] / s)
    out[g] = dict(effect_all=float(d.square().mean().sqrt()), effect_moving=float(mov(d).square().mean().sqrt()), signal_moving=float(one.square().mean().sqrt()))
(R / "true_scale.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out))
