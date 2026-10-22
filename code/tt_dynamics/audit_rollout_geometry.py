"""One-shot source MJX geometry audit; survival 'valid' is NOT a contact criterion.

The existing sources are not rewritten. This reports a reproducible random
subset of frames marked valid in NPZ, then compares those frames against the
2mm generated-S0 geometry criterion and floor-contact distribution.
"""
import argparse
import glob
import json
from pathlib import Path
from collections import Counter

import numpy as np
from diagnose_s0_penetration import audit, expert_params, build_model


def run(args):
    paths = sorted(Path(p) for p in glob.glob(args.data_glob))
    if not paths:
        raise FileNotFoundError(args.data_glob)
    choice, kp, tau, torque = expert_params(args.expert_config)
    rng = np.random.default_rng(args.seed)
    result = {"valid_definition": "rollout survival/time mask; not geometry validity",
              "criteria": "2mm foot/nonfoot/self penetration; foot contact count recorded separately",
              "max_frames_per_shape": args.max_frames_per_shape,
              "sources": []}
    for path in paths:
        with np.load(path) as z:
            mask = z["valid"]
            frames = np.argwhere(mask)
            if not len(frames):
                result["sources"].append({"path": str(path), "error": "no valid frames"})
                continue
            sampled = frames if args.max_frames_per_shape == 0 else frames[
                rng.choice(len(frames), min(args.max_frames_per_shape, len(frames)), replace=False)]
            e = np.asarray(z["E"], float)
            if e.shape != (13,):
                raise ValueError(f"expected fixed E[13] in {path}: {e.shape}")
            model = build_model(choice, kp=kp, tau_scale=tau, torque_limit=torque, uniforms=e)
            qpos, qvel = z["qpos"], z["qvel"]
            collision_ok, penetrations, contacts = 0, [], []
            bad = Counter()
            for b, t in sampled:
                r = audit(model, qpos[b, t], qvel[b, t], tolerance=args.tolerance_mm / 1000.,
                          min_support=0)
                collision_reasons = [reason for reason in r["reasons"] if reason in
                                     ("foot_ground_penetration", "nonfoot_ground_penetration",
                                      "self_penetration")]
                if not collision_reasons:
                    collision_ok += 1
                bad.update(collision_reasons)
                penetrations.append(r["max_foot_penetration_mm"])
                contacts.append(r["floor_contact_count"])
            result["sources"].append({
                "path": str(path), "total_valid_masked_frames": int(len(frames)),
                "sampled_valid_frames": int(len(sampled)),
                "sampled_geometry_ok": int(collision_ok),
                "sampled_geometry_failure_fraction": float(1 - collision_ok / len(sampled)),
                "collision_reasons": dict(bad),
                "max_foot_penetration_mm_p50_p95_max": [
                    float(v) for v in np.percentile(penetrations, [50, 95, 100])],
                "foot_contact_count_hist": dict(Counter(map(str, contacts))),
            })
        print(f"checked: {path}", flush=True)
    result["scope_note"] = ("Finite random sample, unless --max-frames-per-shape=0. "
                            "Does not relabel/remove training episodes or prove TT learnability.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"Saved: {args.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-glob", default="runs/pipeline_v1/wide16_action/rollouts/e*/primitive_seed0.npz")
    p.add_argument("--expert-config", type=Path,
                   default=Path("runs/pipeline_v1/experts_wide16/walkForwardSlow__r00"))
    p.add_argument("--max-frames-per-shape", type=int, default=128,
                   help="0 checks ALL frames, default 128 sampled valid frames per E")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tolerance-mm", type=float, default=2.)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.max_frames_per_shape < 0 or args.tolerance_mm <= 0:
        p.error("invalid frame sample count or collision tolerance")
    run(args)


if __name__ == "__main__":
    main()
