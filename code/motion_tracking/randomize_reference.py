"""Randomize eligible A1 references while keeping the 50 Hz retarget input format."""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pd_replay import ROOT, build_model, foot_lock, inverse_check


EXCLUDED = {"tiltYaw", "tiltYawRaw", "backFlip", "howAreYou", "rollBack", "jumpInPlace",
            "stairsUpDown", "walkToJump", "jumpToWalk", "walkToStand"}
DT = 0.02


def randomize(z, model, rng):
    speed = rng.uniform(0.9, 1.1)
    stride = rng.uniform(0.95, 1.05)
    heading = rng.uniform(-np.pi / 12, np.pi / 12)
    old_dt = float(z["dt"])
    source_t = np.arange(len(z["fix_q"])) * old_dt
    target_t = np.arange(int(np.floor(source_t[-1] / speed / DT)) + 1) * DT
    sample_t = np.minimum(target_t * speed, source_t[-1])

    def sample(x):
        return np.stack([np.interp(sample_t, source_t, x[:, i]) for i in range(x.shape[1])], axis=1)

    q = sample(z["fix_q"])
    base_pos = sample(z["fix_base_pos"])
    base_quat = Slerp(source_t, Rotation.from_quat(z["fix_base_quat"][:, [1, 2, 3, 0]]))(sample_t)
    base_quat = base_quat.as_quat()[:, [3, 0, 1, 2]]
    foot = np.stack([sample(z["fix_foot_pos"][:, leg]) for leg in range(4)], axis=1)
    contact = z["fix_contact"][np.clip(np.rint(sample_t / old_dt).astype(int), 0, len(source_t) - 1)]

    # Change horizontal travel and foot placement together; IK restores reachable, planted feet.
    origin = base_pos[0, :2].copy()
    base_pos[:, :2] = origin + stride * (base_pos[:, :2] - origin)
    foot[:, :, :2] = origin + stride * (foot[:, :, :2] - origin)
    ref = dict(ref_base_pos=base_pos, ref_base_quat=base_quat,
               ref_foot_pos=foot, ref_contact=contact)
    fix = foot_lock(model, dict(q=q, dt=DT), ref)

    rotation = Rotation.from_euler("z", heading)
    center = fix["fix_base_pos"][0].copy()
    for key in ("fix_base_pos", "fix_foot_pos", "fix_foot_target"):
        fix[key] = center + rotation.apply(fix[key] - center)
    quat = Rotation.from_quat(fix["fix_base_quat"][:, [1, 2, 3, 0]])
    fix["fix_base_quat"] = (rotation * quat).as_quat()[:, [3, 0, 1, 2]]
    inv = inverse_check(model, fix, DT)
    assert len(q) == len(fix["fix_q"]) == len(inv["inv_qvel"])
    assert np.isfinite(fix["fix_q"]).all() and np.isfinite(inv["inv_qvel"]).all()
    return fix, inv, dict(speed=speed, stride=stride, heading_rad=heading)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--out", type=Path, default=Path(ROOT) / "artifacts" / "randomized_reference")
    args = parser.parse_args()
    if args.variants < 1:
        parser.error("--variants must be positive")
    src = Path(ROOT) / "artifacts" / "pd_replay"
    names = sorted(p.stem for p in src.glob("*.npz") if p.stem not in EXCLUDED)
    if args.only:
        missing = set(args.only) - set(names)
        if missing:
            parser.error(f"unknown or excluded motions: {sorted(missing)}")
        names = [n for n in names if n in args.only]
    model = build_model()
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        with np.load(src / f"{name}.npz") as z:
            for variant in range(args.variants):
                fix, inv, params = randomize(z, model, rng)
                out = args.out / f"{name}__r{variant:02d}.npz"
                np.savez_compressed(out, **fix, **inv, dt=DT,
                                    source_motion=name, seed=args.seed, **params)
                rows.append(dict(motion=name, variant=variant, frames=len(fix["fix_q"]),
                                 **params, slip=fix["fix_slip"],
                                 foot_err_mm=fix["fix_track_err"] * 1000,
                                 inv_bad_frac=inv["inv_bad_frac"],
                                 source_inv_bad_frac=float(z["inv_bad_frac"]),
                                 joint_limit_frac=float(((fix["fix_q"] <= model.jnt_range[1:, 0] + 1e-3)
                                                         | (fix["fix_q"] >= model.jnt_range[1:, 1] - 1e-3)).any(axis=1).mean())))
                print(f"{out.name}: {len(fix['fix_q'])} frames, "
                      f"slip={fix['fix_slip']:.3f} m/s, inverse bad={inv['inv_bad_frac']:.1%}", flush=True)
    with (args.out / "manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
