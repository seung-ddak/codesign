"""co-design 검사: (1) 형상 E 가 MJX/TT 결과를 바꾸는가 (η² + permutation), (2) joint guidance 가 고른 E 가 좋은가 (백분위)."""
import json
from pathlib import Path
import numpy as np

here = Path(__file__).parent
rng = np.random.default_rng(0)


def eta2(y, g):
    m = y.mean()
    between = sum(len(y[g == k]) * (y[g == k].mean() - m) ** 2 for k in np.unique(g))
    return between / ((y - m) ** 2).sum()


def perm_p(y, g, n=10000):
    e = eta2(y, g)
    return e, float(np.mean([eta2(y, rng.permutation(g)) >= e for _ in range(n)]))


out = {}
for task in ("zigzag", "circle"):
    fx = json.load(open(here / f"fixedE_{task}_4s/summary.json"))
    names, per = fx.get("fixed_e") or None, None
    rs = fx["runs"]["scale_10"]
    n_design = 16
    per = len(rs) // n_design
    g = np.repeat(np.arange(n_design), per)
    ok = np.array([r["mjx_total"] is not None for r in rs])
    mjx = np.array([r["mjx_total"] if r["mjx_total"] is not None else np.nan for r in rs])
    prog = np.array([r["mjx_progress"] if r["mjx_progress"] is not None else np.nan for r in rs])
    pred = np.array([r["pred_total"] for r in rs])
    done = np.array([r["mjx_fell_step"] == -1 if r["mjx_fell_step"] is not None else False for r in rs])
    res = {"valid": int(ok.sum()), "n": len(rs), "done": int(done.sum())}
    for key, y in (("mjx_R", mjx), ("mjx_progress", prog), ("pred_R", pred), ("survive", done.astype(float))):
        m = ~np.isnan(y)
        res[key] = dict(zip(("eta2", "p"), perm_p(y[m], g[m])))
    design_mean = np.array([np.nanmean(mjx[g == k]) for k in range(n_design)])
    jt = json.load(open(here / f"joint_{task}_4s/summary.json"))["runs"]["scale_10"]
    jm = [r["mjx_total"] for r in jt if r["mjx_total"] is not None]
    res["design_mean_R"] = {"min": float(design_mean.min()), "median": float(np.median(design_mean)), "max": float(design_mean.max())}
    res["joint"] = {"valid": len(jm), "done": sum(r["mjx_fell_step"] == -1 for r in jt if r["mjx_total"] is not None),
                    "median_R": float(np.median(jm)), "percentile_vs_designs": float((design_mean < np.median(jm)).mean() * 100),
                    "median_progress": float(np.median([r["mjx_progress"] for r in jt if r["mjx_total"] is not None]))}
    res["fixed_median_R"], res["fixed_median_progress"] = float(np.nanmedian(mjx)), float(np.nanmedian(prog))
    out[task] = res
    print(task, json.dumps(res, ensure_ascii=False, indent=1))
json.dump(out, open(here / "codesign_check.json", "w"), indent=1, ensure_ascii=False)
