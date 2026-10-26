"""Evaluate V7 100-step DDIM experiments using ACTUAL MJX rewards, not TT predictions.

Produces JSON; no success claim without executed, finite MJX trajectories.
Stage failure or missing artifacts is recorded as "not_run", never as success.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def load(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def summarize(rows):
    valid = [r for r in rows if finite(r.get("mjx_total")) and r.get("execution_skipped") is not True
             and isinstance(r.get("initial_physics"), dict)
             and r["initial_physics"].get("physically_valid") is True]
    complete = [r for r in valid if r.get("mjx_fell_step") == -1]
    metrics = {"candidates": len(rows), "valid_mjx": len(valid),
               "completed": len(complete), "completion_rate": len(complete) / len(rows) if rows else 0}
    for key in ("mjx_total", "pred_total", "mjx_progress", "pred_progress", "path_err_m"):
        vals = np.array([r[key] for r in valid if finite(r.get(key))], dtype=float)
        metrics[key + "_median"] = float(np.median(vals)) if len(vals) else None
    gap = [abs(r["pred_total"] - r["mjx_total"]) for r in valid if finite(r.get("pred_total"))]
    metrics["reward_abs_gap_median"] = float(np.median(gap)) if gap else None
    return metrics


def rank_corr(a, b):
    """Spearman rho (average ties), no scipy dependency."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return None
    def ranks(v):
        idx = np.argsort(v, kind="stable")
        out = np.empty(len(v), float)
        s = 0
        while s < len(v):
            e = s + 1
            while e < len(v) and v[idx[e]] == v[idx[s]]:
                e += 1
            out[idx[s:e]] = (s + e - 1) / 2.0
            s = e
        return out
    x, y = ranks(a), ranks(b)
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def eta2_p(values, groups, n=2000, seed=0):
    y, g = np.asarray(values, float), np.asarray(groups)
    good = np.isfinite(y)
    y, g = y[good], g[good]
    if len(y) < 4 or len(np.unique(g)) < 2:
        return {"eta2": None, "permutation_p": None}
    def score(labels):
        total = float(((y - y.mean()) ** 2).sum())
        if total <= 1e-12:
            return 0.0
        between = sum(int((labels == k).sum()) * (y[labels == k].mean() - y.mean()) ** 2
                      for k in np.unique(labels))
        return float(between / total)
    observed = score(g)
    rng = np.random.default_rng(seed)
    p = (1 + sum(score(rng.permutation(g)) >= observed for _ in range(n))) / (n + 1)
    return {"eta2": observed, "permutation_p": float(p)}


def analyze_fixed(fixed_summary, joint_summary, scale):
    tag = f"scale_{scale:g}"
    if fixed_summary is None or joint_summary is None:
        return {"status": "not_run"}
    rows = fixed_summary.get("runs", {}).get(tag, [])
    joint = joint_summary.get("runs", {}).get(tag, [])
    names = fixed_summary.get("fixed_e")
    # Generator may store either list of designs or a nonempty list thereof.
    if not isinstance(names, list) or not names:
        return {"status": "invalid", "reason": "missing fixed_e design identifiers"}
    n_design = len(names)
    if len(rows) < n_design or len(rows) % n_design:
        return {"status": "invalid", "reason": "fixed-E row count not divisible by number of designs"}
    per = len(rows) // n_design
    grouped = np.repeat(np.arange(n_design), per)
    pred = np.array([r.get("pred_total", np.nan) for r in rows], dtype=float)
    mjx = np.array([r.get("mjx_total") if finite(r.get("mjx_total")) else np.nan for r in rows], float)
    pred_means, mjx_means = [], []
    for k in range(n_design):
        ps, ms = pred[grouped == k], mjx[grouped == k]
        pred_means.append(float(np.mean(ps)) if np.isfinite(ps).all() else np.nan)
        mjx_means.append(float(np.mean(ms)) if np.isfinite(ms).all() else np.nan)
    rho = rank_corr(pred_means, mjx_means)
    variation = eta2_p(pred, grouped)
    path = summarize(joint)
    ready = bool(rho is not None and rho > 0.6 and variation["permutation_p"] is not None
                 and variation["permutation_p"] < 0.05)
    return {"status": "evaluated", "fixed": summarize(rows), "joint": path,
            "n_shapes": n_design, "samples_per_shape": per,
            "predicted_reward_by_shape": pred_means,
            "actual_mjx_reward_by_shape": mjx_means,
            "spearman_shape_reward_rho": rho,
            "predicted_reward_e_effect": variation,
            "morphology_gate_pass": ready,
            "morphology_gate_definition": "predicted E effect permutation p<0.05 AND shape-mean Spearman rho>0.6",
            "joint_mjx_reward_better_than_fixed_median": (
                path["mjx_total_median"] > summarize(rows)["mjx_total_median"]
                if finite(path["mjx_total_median"]) and finite(summarize(rows)["mjx_total_median"]) else None)}


def evaluate(root, stage, fixed_scale):
    result = {"stage": stage, "experiment": str(root), "diffusion_steps_requested": 100,
              "status": "audit_only_no_gpu_execution_by_this_script"}
    if stage in ("smoke", "all"):
        d = load(root / "smoke_zigzag_2s/summary.json")
        if d is None:
            result["smoke"] = {"status": "not_run"}
        else:
            result["smoke"] = {"status": "evaluated",
                               "scales": {scale: summarize(rows) for scale, rows in d["runs"].items()}}
            base = result["smoke"]["scales"].get("scale_0", {})
            # Gate is about trajectory agreement, not just total reward.
            err = base.get("path_err_m_median")
            result["smoke"]["dynamics_gate_pass"] = bool(finite(err) and err < 0.15
                                                   and base.get("valid_mjx", 0) > 0)
    if stage in ("codesign", "all"):
        result["codesign"] = {task: analyze_fixed(
            load(root / f"codesign/fixedE_{task}_4s/summary.json"),
            load(root / f"codesign/joint_{task}_4s/summary.json"),
            fixed_scale) for task in ("zigzag", "circle")}
    if stage in ("full", "all"):
        result["full"] = {}
        for task in ("zigzag", "circle"):
            d = load(root / f"full_{task}_10s/summary.json")
            result["full"][task] = {"status": "evaluated",
                                   "scales": {scale: summarize(rows) for scale, rows in d["runs"].items()}} if d else {"status": "not_run"}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=("smoke", "codesign", "full", "all"), default="all")
    p.add_argument("--fixed-scale", type=float, default=10.)
    args = p.parse_args()
    data = evaluate(args.root, args.stage, args.fixed_scale)
    out = args.root / f"audit_{args.stage}.json"
    args.root.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    print("Saved:", out)


if __name__ == "__main__":
    main()
