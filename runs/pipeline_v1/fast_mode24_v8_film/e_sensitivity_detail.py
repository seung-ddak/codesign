"""Group bootstrap CI and static/moving split of the E-swap agreement. Reuses e_sensitivity.py."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "code/tt_dynamics")
import e_sensitivity as es
from rollout_data import S_DIM
from tt_model import EGuidedNet, make_net, rebase, schedule

R = Path("runs/pipeline_v1/fast_mode24_v8_film")
RUNS = dict(v7=("runs/pipeline_v1/fast_mode24_v7_eswap/tt_joint", 1), A=(R / "A/tt_joint", 1), B_w1=(R / "B/tt_joint", 1), B_w2=(R / "B/tt_joint", 2))
GROUPS = ("height", "tilt", "joint", "yaw", "pos", "linvel")
dev = "cuda"

def corr(c, t):
    return float(np.corrcoef(c.ravel(), t.ravel())[0, 1])

def stats(c, t, gid, rng):
    moving_c, moving_t = c - c.mean(1, keepdims=True), t - t.mean(1, keepdims=True)
    ids = np.unique(gid); boot = []
    for _ in range(500):
        take = np.concatenate([np.flatnonzero(gid == g) for g in rng.choice(ids, len(ids))])
        boot.append(corr(c[take], t[take]))
    return dict(all=corr(c, t), ci=[float(v) for v in np.percentile(boot, [2.5, 97.5])],
                static=corr(c.mean(1), t.mean(1)), moving=corr(moving_c, moving_t),
                moving_size=float(moving_c.std() / moving_t.std()))

out = {}
for name, (path, w) in RUNS.items():
    ck = torch.load(Path(path) / "checkpoint.pt", map_location=dev, weights_only=False)
    cfg = ck["args"]; net = make_net(cfg).to(dev); net.load_state_dict(ck["net"]); net.eval()
    if w != 1: net = EGuidedNet(net, w)
    H, clip = cfg["horizon"], cfg["clip"]
    mu, sd = [torch.as_tensor(v, device=dev, dtype=torch.float32) for v in (ck["mu"], ck["sd"])]
    beta, abar = schedule(cfg["diff_steps"], dev)
    eps, pairs = es.load_pairs(ck, 0)
    rows = []
    for (di, bi), (dj, bj) in pairs:
        a, b = es.window(eps[di], bi, 30, H), es.window(eps[dj], bj, 30, H)
        if a is not None and b is not None:
            rows.append((a, b, eps[di]["E"], eps[dj]["E"], f'{eps[di]["source_design"][bi]}/{eps[di]["source_episode"][bi]}'))
    gid = np.array([r[4] for r in rows])
    t_ = lambda c: torch.tensor(np.stack([r[c] for r in rows]), device=dev, dtype=torch.float32)
    xi, xj = rebase(t_(0)), rebase(t_(1)); ei, ej = (t_(2) - .5) / .5, (t_(3) - .5) / .5
    si, sj = ((xi - mu) / sd)[:, 0, :S_DIM], ((xj - mu) / sd)[:, 0, :S_DIM]
    run = lambda s0, e: torch.cat([es.generate(net, s0[i:i + 128], e[i:i + 128], H, mu, sd, beta, abar, clip, 8, dev) for i in range(0, len(e), 128)])
    base = run(si, ei); true = (xj - xi)[:, 1:].cpu().numpy(); rng = np.random.default_rng(0); res = {"pairs": len(rows), "groups": len(set(gid))}
    for cond, gen in dict(E_only=run(si, ej), both=run(sj, ej)).items():
        ch = (gen - base)[:, 1:].cpu().numpy()
        res[cond] = {g: stats(ch[..., es.GROUPS[g][0]:es.GROUPS[g][1]], true[..., es.GROUPS[g][0]:es.GROUPS[g][1]], gid, rng) for g in GROUPS}
    # same E, same S0, other seed: how big is the sampling noise of the mean itself?
    ch_noise = (run(si, ei) - base)[:, 1:].cpu().numpy()
    res["repeat_noise_max"] = float(np.abs(ch_noise).max())
    out[name] = res
    print(name, json.dumps({c: {g: {k: round(v[k], 2) if k != "ci" else [round(x, 2) for x in v[k]] for k in v} for g, v in res[c].items()} for c in ("E_only",)}), flush=True)
(R / "sensitivity/detail_start30.json").write_text(json.dumps(out, indent=1))
print("DETAIL_OK")
