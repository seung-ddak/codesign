"""Small clamp/normalization audit before another FAST 2-second smoke run.

Reads the existing trained TT checkpoint plus saved generated trajectories.
No model retraining, sampling, geometry changes or MJX stepping. This audit
answers ONLY whether the identical initial body heights equal the training
normalization boundary (and whether each generated S0 saturates the sampler clip).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch


def inspect_s0_height(mu, sd, clip, trajectories, *, eps=1e-4):
    mu_z, sd_z = float(mu[2]), float(sd[2])
    if not np.isfinite([mu_z, sd_z, clip]).all() or sd_z <= 0 or clip <= 0:
        raise ValueError("Invalid checkpoint normalization or sampler clipping")
    records = {}
    for name, x in trajectories.items():
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[2] != 54 or not len(x) or not np.isfinite(x).all():
            raise ValueError(f"{name}: expected finite [candidates,T,54] trajectory")
        zs = x[:, 0, 2]
        zn = (zs - mu_z) / sd_z
        records[name] = {
            "initial_heights_m": zs.astype(float).tolist(),
            "initial_height_norm": zn.astype(float).tolist(),
            "lower_clip_hits": np.isclose(zn, -clip, rtol=0, atol=eps).tolist(),
            "upper_clip_hits": np.isclose(zn, clip, rtol=0, atol=eps).tolist(),
        }
    hit_lower = any(any(v["lower_clip_hits"]) for v in records.values())
    every_lower = all(all(v["lower_clip_hits"]) for v in records.values())
    return {
        "normalization_mean_z": mu_z,
        "normalization_half_range_z": sd_z,
        "normalized_clip": float(clip),
        "physical_lower_clip_m": mu_z - clip * sd_z,
        "physical_upper_clip_m": mu_z + clip * sd_z,
        "any_lower_clip_hit": bool(hit_lower),
        "all_initial_heights_at_lower_clip": bool(every_lower),
        "candidates": records,
        "interpretation": (
            "The height appears pinned to the lower sampling boundary; "
            "clipping is observed, but this alone does not establish WHY the "
            "model predicted low height. Do not increase the clip or shift the "
            "body as an unverified fix."
            if hit_lower else
            "The sampled S0 heights are not pinned to the lower clamp. "
            "Investigate joint E, body height and joint-angle feasibility."
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--smoke-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eps", type=float, default=1e-4)
    args = p.parse_args()
    if args.eps <= 0:
        p.error("--eps must be positive")
    result = {"status": "FAIL", "stage": "S0 height clipping"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        ck = torch.load(args.ckpt / "checkpoint.pt", map_location="cpu", weights_only=False)
        clip = ck["args"]["clip"]
        trajectories = {}
        for scale in ("0", "1"):
            with np.load(args.smoke_dir / f"scale_{scale}.npz") as d:
                trajectories[f"scale_{scale}"] = d["pred"].copy()
        result = inspect_s0_height(ck["mu"], ck["sd"], clip, trajectories, eps=args.eps)
        result.update(status="PASS", stage="S0 height clipping")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    print(f"{result['status']}: {result['stage']}", flush=True)
    print(result.get("interpretation", result.get("error", "")), flush=True)
    print(f"Details: {args.out}", flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
