"""Compare old/new TT action sensitivity with MJX on held-out paired starts.

Run from codesign/: python code/tt_dynamics/eval_action_pairs.py --data <pairs.npz> \
    --new <new-run> --old runs/pipeline_v1/tt/h32_try6 --out <result.json>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rollout_data import features
from tt_model import Net, condition_masks, schedule

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--new", type=Path, required=True)
parser.add_argument("--old", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--seeds", type=int, default=4)
parser.add_argument("--split", choices=("train", "val", "test"), help="evaluate the given dataset split instead of checkpoint validation rows")
parser.add_argument("--plot", type=Path)
args = parser.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
if args.seeds < 1:
    parser.error("--seeds must be positive")

with np.load(args.data) as z:
    q, v, c, a, valid, E = (z[k] for k in ("qpos", "qvel", "contact", "target_q", "valid", "E"))
    group = z["group_id"]
    variant = z["variant"]
    labels = z["split"] if "split" in z else None
    episodes = z["source_episode"] if "source_episode" in z else None
H = a.shape[1]
assert len(a) % 3 == 0 and np.array_equal(group.reshape(-1, 3)[:, 0], group.reshape(-1, 3)[:, 2])
assert np.array_equal(variant.reshape(-1, 3)[0], ["orig", "swap", "offset"])

cknew = torch.load(args.new / "checkpoint.pt", map_location="cpu", weights_only=False)
if args.split:
    if labels is None:
        raise ValueError("--split requires split labels in the paired NPZ")
    groups = np.unique(group[labels == args.split])
else:
    val_rows = np.asarray(cknew["val"])
    if len(val_rows) == 0 or val_rows.max() >= len(a):
        raise ValueError("new checkpoint validation indices do not match paired data")
    groups = np.unique(group[val_rows])
if not len(groups):
    raise ValueError("no starts in selected evaluation split")
select = np.flatnonzero(np.isin(group, groups))
assert len(select) == len(groups) * 3, "validation split broke a counterfactual group"
if labels is not None and args.split and not np.all(labels[select] == args.split):
    raise ValueError("a start appears in multiple dataset splits")
group, episodes = group[select], episodes[select] if episodes is not None else None
q, v, c, a, valid = (arr[select] for arr in (q, v, c, a, valid))
x = np.concatenate([np.stack([features(qi, vi, ci) for qi, vi, ci in zip(q, v, c)]), a], -1)
mask = valid.reshape(-1, 3, H).all((1, 2))
if not mask.any():
    raise ValueError("no held-out groups survive all three variants")


def metrics(states):
    end = states[:, -1]
    yaw = np.degrees(np.arctan2(end[:, 4], end[:, 3]))
    return {"dx_m": end[:, 0], "dy_m": end[:, 1], "yaw_deg": yaw,
            "height_m": end[:, 2], "joint_rad": end[:, 14:26]}


truth = metrics(x)


@torch.no_grad()
def predict(ck):
    config = ck["args"]
    if config["horizon"] != H or config.get("pose", "abs") != "abs":
        raise ValueError("checkpoint must use the paired horizon and abs pose")
    net = Net(H, config["d_model"], config["layers"]).to(dev).eval()
    net.load_state_dict(ck["net"])
    mu, sd = (torch.as_tensor(ck[k], device=dev) for k in ("mu", "sd"))
    emu, esd = ((0.5, 0.5) if config["norm"] == "minmax" else (0.5, 0.25))
    e = torch.as_tensor(np.broadcast_to(E, (len(x), len(E))).copy(), device=dev)
    e = (e - emu) / esd
    x0 = (torch.as_tensor(x, device=dev) - mu) / sd
    known, ke = condition_masks(torch.ones(len(x), dtype=torch.long, device=dev),
                                torch.ones(len(x), H, dtype=torch.bool, device=dev))
    beta, abar = schedule(config["diff_steps"], dev)
    samples = []
    for seed in range(args.seeds):
        # Identical noise within each three-way group isolates the action effect.
        noise = torch.randn((len(groups), H, x.shape[-1]), generator=torch.Generator(dev).manual_seed(seed), device=dev)
        xt = noise.repeat_interleave(3, 0)
        for k in reversed(range(len(beta))):
            ab, prev = abar[k], (abar[k - 1] if k else torch.ones((), device=dev))
            xt = torch.where(known > 0, x0, xt)
            ex, _ = net(xt, e, known, ke, torch.full((len(x),), k, device=dev, dtype=torch.float32),
                        torch.zeros(len(x), H, dtype=torch.bool, device=dev))
            denoised = ((xt - (1 - ab).sqrt() * ex) / ab.sqrt()).clamp(-config["clip"], config["clip"])
            xt = prev.sqrt() * denoised + (1 - prev).sqrt() * (xt - ab.sqrt() * denoised) / (1 - ab).sqrt()
        samples.append((torch.where(known > 0, x0, xt) * sd + mu).cpu().numpy())
    return metrics(np.mean(samples, axis=0)), [metrics(sample) for sample in samples]


def score(pred):
    result = {}
    for name, actual in truth.items():
        result[name] = {}
        for j, label in ((1, "swap"), (2, "offset")):
            delta = (actual.reshape(-1, 3, *actual.shape[1:])[:, j] - actual.reshape(-1, 3, *actual.shape[1:])[:, 0])[mask]
            estimate = (pred[name].reshape(-1, 3, *actual.shape[1:])[:, j] -
                        pred[name].reshape(-1, 3, *actual.shape[1:])[:, 0])[mask]
            if name == "yaw_deg":
                delta = (delta + 180) % 360 - 180
                estimate = (estimate + 180) % 360 - 180
            denom = np.square(delta).sum()
            result[name][label] = {"mjx_delta_rms": float(np.sqrt(np.square(delta).mean())),
                                   "tt_delta_rms": float(np.sqrt(np.square(estimate).mean())),
                                   "slope": float((delta * estimate).sum() / denom) if denom > 1e-12 else None,
                                   "delta_mae": float(np.abs(delta - estimate).mean())}
    return result


old = torch.load(args.old / "checkpoint.pt", map_location="cpu", weights_only=False)
old_mean, old_seeds = predict(old)
new_mean, new_seeds = predict(cknew)
result = {"data": str(args.data), "split": args.split or "checkpoint_val", "held_out_starts": int(mask.sum()),
          "new": str(args.new), "old": str(args.old), "old_scores": score(old_mean), "new_scores": score(new_mean),
          "seed_scores": [{"seed": seed, "old": score(old_seeds[seed]), "new": score(new_seeds[seed])}
                          for seed in range(args.seeds)], "per_start": []}
for i in np.flatnonzero(mask):
    row = {"group_id": int(group[3 * i]), "source_episode": int(episodes[3 * i]) if episodes is not None else None,
           "responses": {}}
    for name, actual in truth.items():
        row["responses"][name] = {}
        for j, label in ((1, "swap"), (2, "offset")):
            values = [actual[3 * i + j] - actual[3 * i], old_mean[name][3 * i + j] - old_mean[name][3 * i],
                      new_mean[name][3 * i + j] - new_mean[name][3 * i]]
            if name == "yaw_deg":
                values = [(value + 180) % 360 - 180 for value in values]
            row["responses"][name][label] = {k: np.asarray(value).tolist() for k, value in zip(("mjx", "A", "B"), values)}
    result["per_start"].append(row)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
if args.plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = [("yaw_deg", "swap"), ("yaw_deg", "offset"), ("joint_rad", "swap"), ("joint_rad", "offset"),
              ("dx_m", "swap"), ("dy_m", "offset")]
    labels_plot = [f"{m} / {a}" for m, a in chosen]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    xx = np.arange(len(chosen))
    for axis, key, reference in ((ax[0], "slope", 1), (ax[1], "delta_mae", 1)):
        for offset, (title, scores) in ((-0.19, ("A orig", result["old_scores"])),
                                         (0.19, ("B intervention", result["new_scores"]))):
            values = [scores[m][a][key] / max(result["old_scores"][m][a][key], 1e-12) if key == "delta_mae" else scores[m][a][key]
                      for m, a in chosen]
            axis.bar(xx + offset, values, width=0.38, label=title)
        if reference is not None:
            axis.axhline(reference, color="black", linestyle="--", linewidth=1)
        axis.set_xticks(xx, labels_plot, rotation=35, ha="right")
        axis.set_title("ΔMAE / A (lower is better)" if key == "delta_mae" else "response slope (1 is ideal)")
        axis.legend()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.plot, dpi=160)
print(f"{result['held_out_starts']} starts -> {args.out}" + (f", {args.plot}" if args.plot else ""))
