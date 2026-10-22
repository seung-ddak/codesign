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
    # Hash PHYSICAL source episode identity, not the morphology-specific file
    # path. Replicas across e000/e001/... must carry the same split label.
    # Existing group_id is kept; source_episode wins over fallback row index.
    origin = data["source_episode"] if "source_episode" in data else np.arange(n)
    if "group_id" in data:
        gid = np.asarray(data["group_id"], dtype=np.int64)
    else:
        gid = np.array([
            int.from_bytes(hashlib.sha256(f"{path.stem}:{int(b)}".encode()).digest()[:8],
                           "little") % (2**63 - 1) for b in origin
        ], dtype=np.int64)
        data["group_id"] = gid
    # Shared ID-based split assignment is independent of file ordering/count.
    score = np.array([
        int.from_bytes(hashlib.sha256(f"{args.seed}:{int(g)}".encode()).digest()[:8],
                       "little") / float(2**64) for g in gid
    ])
    split = np.full(n, "train", dtype="<U5")
    split[score < args.test_frac] = "test"
    split[(score >= args.test_frac) & (score < args.test_frac + args.val_frac)] = "val"
    data["split"] = split
    data["source_episode"] = np.asarray(origin, dtype=np.int32)
    val_n, test_n = int((split == "val").sum()), int((split == "test").sum())
    if not (split == "train").any():
        raise ValueError(f"{path}: deterministic grouped split leaves no train episodes")
    digest = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    dest = args.out / f"general_{path.stem}_{digest}.npz"
    if dest.exists():
        raise FileExistsError(dest)
    tmp = dest.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(dest)
    print(f"{path} -> {dest}: {n} episodes, {sum(split == 'train')} train / {val_n} val / {test_n} test")
