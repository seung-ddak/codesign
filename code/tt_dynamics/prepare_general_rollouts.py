"""Prepare existing expert MJX rollouts as optional general-motion TT training data.

Runs offline on NPZ files; does not execute an expert or regenerate dynamics.
Use distinct source episodes for train/val/test, never split contiguous
windows from the same episode. The source must contain valid, qpos, qvel,
contact, target_q, E and either motions/clip or variant.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--sources", nargs="+", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--seed", type=int, default=20261022)
p.add_argument("--val-frac", type=float, default=0.15)
p.add_argument("--test-frac", type=float, default=0.15)
args = p.parse_args()
if not (0 <= args.val_frac < 1 and 0 <= args.test_frac < 1 and args.val_frac + args.test_frac < 1):
    p.error("split fractions must be nonnegative and sum to less than 1")
args.out.mkdir(parents=True, exist_ok=True)
for path in args.sources:
    with np.load(path, allow_pickle=False) as z:
        required = {"valid", "qpos", "qvel", "contact", "target_q", "E"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}")
        if "motions" not in z.files and "variant" not in z.files:
            raise ValueError(f"{path}: requires motions/clip or variant for rollout_data.load")
        if "motions" in z.files and "clip" not in z.files:
            raise ValueError(f"{path}: motions without clip")
        data = {k: z[k] for k in z.files}
    n = len(data["valid"])
    if n < 3:
        raise ValueError(f"{path}: at least three episodes needed for split")
    if data["target_q"].shape[:2] != data["valid"].shape or data["target_q"].shape[-1] != 12:
        raise ValueError(f"{path}: inconsistent action dimensions")
    E = data["E"]
    if E.shape != (13,) or not np.isfinite(E).all():
        raise ValueError(f"{path}: expected one 13-D E per source NPZ")
    # A fixed source-path hash gives stable splits independent of input ordering.
    h = hashlib.sha256((str(path.resolve()) + ":" + str(args.seed)).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
    order = rng.permutation(n)
    val_n = max(1, round(n * args.val_frac)) if args.val_frac else 0
    test_n = max(1, round(n * args.test_frac)) if args.test_frac else 0
    if val_n + test_n >= n:
        raise ValueError(f"{path}: split leaves no training episodes")
    split = np.full(n, "train", dtype="<U5")
    split[order[:val_n]] = "val"
    split[order[val_n:val_n + test_n]] = "test"
    data["split"] = split
    data["source_episode"] = np.arange(n, dtype=np.int32)
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    dest = args.out / f"general_{path.stem}_{digest}.npz"
    if dest.exists():
        raise FileExistsError(dest)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(dest)
    print(f"{path} -> {dest}: {n} episodes, {sum(split == 'train')} train / {val_n} val / {test_n} test")
