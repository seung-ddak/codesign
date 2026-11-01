"""Evaluate all three models on held-out PD starts, with the historical Mode-2 A-pinning diagnostic."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code/tt_dynamics"))
from e_sensitivity import GROUPS
from rollout_data import S_DIM, load, source_group_keys
from tt_model import condition_masks, make_net, rebase, schedule


@torch.no_grad()
def predict(net, s0, e, action, mu, beta, abar, clip):
    n, horizon = len(e), 32
    dev = s0.device
    mask = torch.ones(n, horizon, dtype=torch.bool, device=dev)
    kx, ke = condition_masks(torch.full((n,), 2, device=dev), mask)
    known = torch.zeros(n, horizon, mu.numel(), device=dev)
    known[:, 0, :S_DIM] = s0
    total = 0
    for seed in range(8):
        generator = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(known.shape, device=dev, generator=generator)
        noise = torch.randn(known.shape, device=dev, generator=generator)
        for k in reversed(range(len(beta))):
            ab, previous = abar[k], abar[k - 1] if k else torch.ones((), device=dev)
            if action is not None:
                x = x.clone()
                x[..., S_DIM:] = ab.sqrt() * action + (1 - ab).sqrt() * noise[..., S_DIM:]
            eps = net(torch.where(kx > 0, known, x), e, kx, ke,
                      torch.full((n,), float(k), device=dev), ~mask)[0]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-clip, clip)
            x = previous.sqrt() * x0 + (1 - previous).sqrt() * (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        total = total + torch.where(kx > 0, known, x0)
    return total / 8


def correlation(a, b):
    a, b = a.flatten().cpu().numpy(), b.flatten().cpu().numpy()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else None


def main():
    conditions = json.loads((RUN / "conditions.json").read_text())
    pd_ck = torch.load(RUN / "tt_joint/checkpoint.pt", map_location="cpu", weights_only=False)
    paths = pd_ck["args"]["data"]
    x, e, mask, _, clips = load(paths, 500)
    groups = []
    for path in paths:
        with np.load(path) as data:
            keys = source_group_keys(path, data, length=len(data["valid"]))
            groups += [keys[i] for i, row in enumerate(data["valid"]) if row.sum() >= 2]
    val = np.asarray(pd_ck["val"])
    train = np.setdiff1d(np.arange(len(x)), val)
    assert not set(np.asarray(groups)[train]) & set(np.asarray(groups)[val])
    metric_ck = torch.load(Path(conditions["metric_reference"]) / "checkpoint.pt", map_location="cpu", weights_only=False)
    metric_scale = torch.as_tensor(metric_ck["sd"], device="cuda", dtype=torch.float32)
    models = {**conditions["baseline_models"], "pd": str(RUN / "tt_joint")}
    result = {"_conditions": dict(validation_source_groups=len(set(np.asarray(groups)[val])),
                                  validation_episodes=len(val), seeds=8, window=32, starts=[0, 30],
                                  metric_reference=conditions["metric_reference"],
                                  wrong_action_donor="different motion, same held-out PD split",
                                  wrong_E_donor="different morphology, same held-out PD split",
                                  interpretation="Native PD data have morphology-specific A; this is not a same-A cross-morphology contrast.")}
    for name, directory in models.items():
        ck = torch.load(Path(directory) / "checkpoint.pt", map_location="cpu", weights_only=False)
        cfg = ck["args"]
        assert cfg["horizon"] == 32 and cfg["modes"] == [2, 4]
        net = make_net(cfg).cuda()
        net.load_state_dict(ck["net"])
        net.eval()
        mu, sd = [torch.as_tensor(ck[k], device="cuda", dtype=torch.float32) for k in ("mu", "sd")]
        beta, abar = schedule(cfg["diff_steps"], "cuda")
        result[name] = {}
        for start in (0, 30):
            indices = val[mask[val].sum(1) >= start + 32]
            assert len(indices)
            truth = rebase(torch.as_tensor(x[indices, start:start + 32], device="cuda"))
            physical_e = e[indices]
            labels = clips[indices, start]
            rng = np.random.default_rng(0)
            wrong_e = np.array([rng.choice(np.flatnonzero(np.linalg.norm(physical_e - ee, axis=1) > 1e-6)) for ee in physical_e])
            wrong_a = np.array([rng.choice(np.flatnonzero(labels != label)) for label in labels])
            norm = (truth - mu) / sd
            en = torch.as_tensor((physical_e - .5) / .5, device="cuda", dtype=torch.float32)
            action = norm[..., S_DIM:]
            assert torch.all((action - action[wrong_a]).abs().amax((1, 2)) > 1e-6)
            predictions = {}
            for variant, ee, aa in (("free", en, None), ("pin", en, action),
                                    ("wrongE", en[wrong_e], action), ("wrongA", en, action[wrong_a])):
                predictions[variant] = torch.cat([
                    predict(net, norm[i:i + 128, 0, :S_DIM], ee[i:i + 128],
                            None if aa is None else aa[i:i + 128], mu, beta, abar, cfg["clip"]) * sd + mu
                    for i in range(0, len(indices), 128)
                ])[:, 1:]
            rows = {"episodes": len(indices)}
            for group in ("pos", "height", "yaw", "tilt", "linvel", "joint"):
                lo, hi = GROUPS[group]
                target = truth[:, 1:, lo:hi] / metric_scale[lo:hi]
                moving = lambda a: a - a.mean(1, keepdim=True)
                scores = {}
                for variant, pred in predictions.items():
                    value = pred[..., lo:hi] / metric_scale[lo:hi]
                    scores[variant + "_rmse"] = float((value - target).square().mean().sqrt())
                    scores[variant + "_moving"] = float((moving(value) - moving(target)).square().mean().sqrt())
                    scores[variant + "_track"] = correlation(moving(value), moving(target))
                rows[group] = scores
            result[name][f"start{start}"] = rows
            print(name, start, len(indices), {g: rows[g]["pin_track"] for g in ("height", "tilt", "linvel")}, flush=True)
        del net, ck
        torch.cuda.empty_cache()
    (RUN / "evaluation/given_a_native.json").write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
