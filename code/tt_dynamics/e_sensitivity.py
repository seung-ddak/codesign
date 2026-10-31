"""Does generated motion follow E? Same-A pairs from held-out E-swap data; no training, no MJX.

For each pair (i, j) sample Mode-2 trajectories with E and S0 of i, then swap E only,
S0 only, or both. Compare the mean predicted change with the true change j - i.
Run from codesign/. Output = JSON at --out.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rollout_data import S_DIM, features
from tt_model import EGuidedNet, condition_masks, make_net, rebase, schedule

GROUPS = dict(pos=(0, 2), height=(2, 3), yaw=(3, 5), tilt=(5, 8), linvel=(8, 11), angvel=(11, 14),
              joint=(14, 26), jointvel=(26, 38), contact=(38, 42))
EP = 500


def load_pairs(ck, seed, per_group=12):
    """Pairs of held-out E-swap episodes sharing one action sequence."""
    val, offset, episodes, groups = set(np.asarray(ck["val"]).tolist()), 0, [], {}
    for path in ck["args"]["data"]:
        with np.load(path, allow_pickle=True) as z:
            keep = [b for b in range(len(z["valid"])) if min(int(z["valid"][b].sum()), EP) >= 2]
            if "source_design" in z.files:
                d = {k: z[k] for k in ("qpos", "qvel", "contact", "target_q", "valid", "E", "source_design", "source_episode")}
                episodes.append(d)
                for i, b in enumerate(keep):
                    if offset + i in val:
                        groups.setdefault((str(d["source_design"][b]), int(d["source_episode"][b])), []).append((len(episodes) - 1, b))
            offset += len(keep)
    rng = np.random.default_rng(seed)
    pairs = [tuple(rng.choice(len(v), 2, replace=False)) + (v,) for v in groups.values() if len(v) > 1 for _ in range(per_group)]
    return episodes, [(v[a], v[b]) for a, b, v in pairs]


def window(d, b, start, H):
    L = int(d["valid"][b].sum())
    if L < start + H:
        return None
    return np.c_[features(d["qpos"][b, :L], d["qvel"][b, :L], d["contact"][b, :L]), d["target_q"][b, :L]][start:start + H]


@torch.no_grad()
def generate(net, s0, e, H, mu, sd, beta, abar, clip, seeds, dev):
    """Mean DDIM Mode-2 trajectory over seeds; the same seeds for every condition."""
    n = len(e)
    mask = torch.ones(n, H, dtype=torch.bool, device=dev)
    kx, ke = condition_masks(torch.full((n,), 2, device=dev), mask)
    known = torch.zeros(n, H, mu.numel(), device=dev)
    known[:, 0, :S_DIM] = s0
    total = 0
    for seed in range(seeds):
        x = torch.randn(known.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(seed))
        for k in reversed(range(len(beta))):
            ab, prev = abar[k], abar[k - 1] if k else torch.ones((), device=dev)
            eps = net(torch.where(kx > 0, known, x), e, kx, ke, torch.full((n,), float(k), device=dev), ~mask)[0]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-clip, clip)
            x = prev.sqrt() * x0 + (1 - prev).sqrt() * (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        total = total + torch.where(kx > 0, known, x0)
    return total / seeds * sd + mu


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--e-cfg", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()
    dev = "cuda"
    ck = torch.load(args.ckpt / "checkpoint.pt", map_location=dev, weights_only=False)
    cfg = ck["args"]
    net = make_net(cfg).to(dev)
    net.load_state_dict(ck["net"])
    net.eval()
    if args.e_cfg != 1:
        net = EGuidedNet(net, args.e_cfg)
    H, clip = cfg["horizon"], cfg["clip"]
    mu, sd = [torch.as_tensor(v, device=dev, dtype=torch.float32) for v in (ck["mu"], ck["sd"])]
    beta, abar = schedule(cfg["diff_steps"], dev)
    episodes, pairs = load_pairs(ck, 0)
    out = dict(checkpoint=str(args.ckpt), e_cfg=args.e_cfg, seeds=args.seeds)
    for start in (0, 30):
        rows = [(window(episodes[di], bi, start, H), window(episodes[dj], bj, start, H), episodes[di]["E"], episodes[dj]["E"])
                for (di, bi), (dj, bj) in pairs]
        rows = [r for r in rows if r[0] is not None and r[1] is not None]
        xi, xj = [rebase(torch.tensor(np.stack([r[c] for r in rows]), device=dev, dtype=torch.float32)) for c in (0, 1)]
        ei, ej = [(torch.tensor(np.stack([r[c] for r in rows]), device=dev, dtype=torch.float32) - .5) / .5 for c in (2, 3)]
        si, sj = [((x - mu) / sd)[:, 0, :S_DIM] for x in (xi, xj)]
        run = lambda s0, e: torch.cat([generate(net, s0[i:i + 128], e[i:i + 128], H, mu, sd, beta, abar, clip, args.seeds, dev)
                                       for i in range(0, len(e), 128)])
        base = run(si, ei)
        true = (xj - xi)[:, 1:]
        res = dict(pairs=len(rows))
        for name, gen in dict(E_only=run(si, ej), S0_only=run(sj, ei), both=run(sj, ej)).items():
            change = (gen - base)[:, 1:]
            res[name] = {}
            for group, (lo, hi) in GROUPS.items():
                c, t = change[..., lo:hi].flatten().cpu().numpy(), true[..., lo:hi].flatten().cpu().numpy()
                res[name][group] = dict(agreement=float(np.corrcoef(c, t)[0, 1]), slope=float(c @ t / (t @ t)),
                                        size_ratio=float(np.sqrt((c ** 2).mean() / (t ** 2).mean())))
        res["true_change_rms"] = {g: float(true[..., lo:hi].square().mean().sqrt()) for g, (lo, hi) in GROUPS.items()}
        out[f"start{start}"] = res
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {n: {g: round(v["agreement"], 2) for g, v in out[k][n].items()} for n in ("E_only", "S0_only", "both")}
                      for k in ("start0", "start30")}))


if __name__ == "__main__":
    main()
