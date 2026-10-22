"""Episode-disjoint A/B manifests and datasets for the e000 action intervention.

Run from codesign/:
  python code/tt_dynamics/prepare_ablation.py --source runs/pipeline_v1/action_pairs/e000_seed0.npz --out-dir runs/pipeline_v1/action_ablation
  # Re-run MJX swaps with collect_action_pairs.py --starts-from ... --manifest ...
  python code/tt_dynamics/prepare_ablation.py --source runs/pipeline_v1/action_pairs/e000_seed0.npz --out-dir runs/pipeline_v1/action_ablation --paired runs/pipeline_v1/action_ablation/paired.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--out-dir", type=Path, required=True)
parser.add_argument("--paired", type=Path)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--episode-split", type=Path, help="shared episode-to-split mapping across morphologies")
args = parser.parse_args()
args.out_dir.mkdir(parents=True, exist_ok=True)
manifest_path = args.out_dir / "split_manifest.json"

with np.load(args.source) as z:
    ids, episodes, times = (z[k][::3] for k in ("group_id", "source_episode", "source_t"))
    assert len(ids) * 3 == len(z["group_id"])
    assert np.array_equal(z["group_id"].reshape(-1, 3), ids[:, None] * np.ones((1, 3), int))
unique = np.unique(episodes)
if len(unique) < 7:
    raise ValueError("too few source episodes for train/val/test split")
if args.episode_split:
    shared = json.loads(args.episode_split.read_text())["episodes"]
    if set(shared.values()) - {"train", "val", "test"} or any(str(int(e)) not in shared for e in unique):
        raise ValueError("invalid or incomplete shared episode split")
    episode_split = {int(e): shared[str(int(e))] for e in unique}
else:
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(unique)
    n_train, n_val = round(0.70 * len(unique)), round(0.15 * len(unique))
    episode_split = {int(e): ("train" if i < n_train else "val" if i < n_train + n_val else "test")
                     for i, e in enumerate(shuffled)}
splits = np.array([episode_split[int(e)] for e in episodes])
donor = np.empty(len(ids), int)
for split in ("train", "val", "test"):
    idx = np.flatnonzero(splits == split)
    if len(idx) < 2:
        raise ValueError(f"{split} has fewer than two starts")
    donor[idx] = ids[np.roll(idx, 1)]
rows = [dict(group_id=int(g), source_episode=int(e), source_t=int(t), split=str(s),
             swap_donor_group_id=int(d)) for g, e, t, s, d in zip(ids, episodes, times, splits, donor)]
manifest = dict(seed=args.seed, source=str(args.source), episodes={str(k): v for k, v in episode_split.items()}, starts=rows)
if args.episode_split:
    manifest["shared_episode_split"] = str(args.episode_split)
if manifest_path.exists():
    if json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"existing manifest differs: {manifest_path}")
else:
    manifest_path.write_text(json.dumps(manifest, indent=2))

if args.paired is None:
    print(f"{manifest_path}: {len(unique)} episodes, {len(ids)} starts, "
          + ", ".join(f"{s}={int((splits == s).sum())}" for s in ("train", "val", "test")))
    raise SystemExit

with np.load(args.paired) as z:
    data = {k: z[k] for k in z.files}
assert np.array_equal(data["group_id"][::3], ids)
assert np.array_equal(data["split"].reshape(-1, 3), np.repeat(splits[:, None], 3, axis=1))
assert np.array_equal(data["swap_donor_group_id"][::3], donor)
lookup = {int(g): i for i, g in enumerate(ids)}
assert all(splits[lookup[int(d)]] == splits[i] and int(d) != int(ids[i]) for i, d in enumerate(donor))
assert np.array_equal(data["target_q"][1::3], data["target_q"][np.array([3 * lookup[int(d)] for d in donor])])
assert all(not (args.out_dir / name).exists() for name in ("model_a_orig.npz", "model_b_all.npz")), "output exists"
for name, selected in (("model_a_orig.npz", data["variant"] == "orig"),
                       ("model_b_all.npz", np.ones(len(data["variant"]), bool))):
    out = {k: (v[selected] if v.ndim and len(v) == len(selected) else v) for k, v in data.items()}
    np.savez_compressed(args.out_dir / name, **out)
    print(f"{name}: {int(selected.sum())} trajectories")
