"""Pairwise D4 evaluation with shared, physically valid starts and matched foot contact.

Each morphology pair uses its own held-out candidates and validity test (no
three-morphology intersection). Within every retained group, qpos0/qvel0/A are
identical. Additional TT training is not performed.

Run from codesign/:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ../transformer-transformer/.venv/bin/python code/tt_dynamics/d4_pairwise.py \
    --out runs/pipeline_v1/diag/D4_pairwise
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT.parent / "transformer-transformer"), str(ROOT / "code/motion_tracking")]

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch
from mujoco import mjx
from omegaconf import OmegaConf

from pd_replay import build_model
from rollout_data import E_DIM, S_DIM, features
from t2.mjx_env.base_env import make_data, step
from track_env import TrackMjxEnv
from tt_model import Net, condition_masks, schedule

SHAPES = ("e000", "e001", "e002")
PAIRS = (("e000", "e001"), ("e000", "e002"), ("e001", "e002"))
H, PENETRATION_LIMIT = 32, 0.005

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", type=Path, default=ROOT / "runs/pipeline_v1/tt/h32_multishape_B")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--starts-per-source", type=int, default=192)
parser.add_argument("--pool-factor", type=int, default=8)
parser.add_argument("--tt-seeds", type=int, default=8)
parser.add_argument("--seed", type=int, default=20261020)
parser.add_argument("--manifests", type=Path, nargs=3, default=[
    ROOT / "runs/pipeline_v1/action_ablation/split_manifest.json",
    ROOT / "runs/pipeline_v1/action_ablation/e001/split_manifest.json",
    ROOT / "runs/pipeline_v1/action_ablation/e002/split_manifest.json",
])
args = parser.parse_args()
if args.starts_per_source < 1 or args.pool_factor < 1 or args.tt_seeds < 1:
    parser.error("starts-per-source, pool-factor, and tt-seeds must be positive")
if args.out.exists() and any(args.out.iterdir()):
    raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.out}")
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Frozen mode-1 checkpoint B.
checkpoint_path = args.ckpt / "checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint["args"]
if config["horizon"] != H or config.get("pose", "abs") != "abs" or config.get("modes") != [1]:
    raise ValueError("checkpoint must be the trained multi-shape B, mode 1, abs pose, horizon 32")
net = Net(H, config["d_model"], config["layers"]).to(device).eval()
net.load_state_dict(checkpoint["net"])
net.requires_grad_(False)
mu, sd = (torch.as_tensor(checkpoint[k], device=device) for k in ("mu", "sd"))
if torch.any(sd == 0):
    raise ValueError("checkpoint has zero normalization scale")
E_MU, E_SD = ((0.5, 0.5) if config["norm"] == "minmax" else (0.5, 0.25))
beta, abar = schedule(config["diff_steps"], device)

# Existing held-out source episodes and frozen rollout windows.
data, test_episodes = {}, {}
for shape, manifest_path in zip(SHAPES, args.manifests):
    manifest = json.loads(manifest_path.read_text())
    test_episodes[shape] = sorted(int(k) for k, v in manifest["episodes"].items() if v == "test")
    data[shape] = np.load(ROOT / f"runs/pipeline_v1/rollouts/{shape}/seq_seed0.npz")
    if not test_episodes[shape] or float(data[shape]["dt"]) != 0.02:
        raise ValueError(f"missing test episodes or unexpected control period for {shape}")

# Same tracking model/controller settings, with each design's existing E.
expert = ROOT / "runs/pipeline_v1/experts/walkForwardSlow__r00"
configs = sorted(expert.glob("wandb/offline-run-*/files/config.pkl"))
if not configs:
    raise FileNotFoundError(expert)
with configs[-1].open("rb") as f:
    expert_cfg = pickle.load(f)
env_cfg = OmegaConf.to_container(expert_cfg.eval_env, resolve=True)
env_cfg.pop("_target_", None)
choice = expert_cfg.get("choice")
envs = {}
for shape in SHAPES:
    E = np.asarray(data[shape]["E"], np.float32)
    cfg = dict(env_cfg)
    cfg["reference_dir"] = str(ROOT / f"runs/pipeline_v1/retargeting/{shape}/{choice}")
    cfg["clips"] = ["walkForwardSlow__r00"]
    envs[shape] = TrackMjxEnv(
        mj_model_fn=lambda E=E: build_model(choice, kp=expert_cfg.get("kp"), tau_scale=expert_cfg.get("tau_scale"),
                                             torque_limit=expert_cfg.get("torque_limit"), uniforms=E), **cfg)


def contact_and_penetration(m, d, feet, floor):
    contact = np.zeros(len(feet), bool)
    foot_ids = {int(g): i for i, g in enumerate(feet)}
    max_pen = 0.0
    for c in d.contact[:d.ncon]:
        pair = {int(c.geom1), int(c.geom2)}
        if floor in pair:
            other = next(iter(pair - {floor}))
            if other in foot_ids and c.dist <= 1e-3:
                contact[foot_ids[other]] = True
                continue
        max_pen = max(max_pen, max(0.0, -float(c.dist)))
    return contact, max_pen


def audit(shape, qpos, qvel, targets, low, high):
    env = envs[shape]
    model, d = env.mj_model, mujoco.MjData(env.mj_model)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(targets).all())
    quat_norm = np.linalg.norm(qpos[3:7])
    joint_bad = any(model.jnt_limited[j] and not (model.jnt_range[j, 0] - 1e-3 <= qpos[model.jnt_qposadr[j]] <= model.jnt_range[j, 1] + 1e-3)
                    for j in range(model.njnt))
    try:
        mujoco.mj_forward(model, d)
        feet = np.asarray(env._foot_ids)
        floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        contact, pen = contact_and_penetration(model, d, feet, floor)
        finite &= np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    except (mujoco.FatalError, ValueError):
        contact, pen, finite = np.zeros(4, bool), float("inf"), False
    valid = finite and abs(quat_norm - 1) <= 1e-3 and not joint_bad and pen <= PENETRATION_LIMIT
    valid &= bool(np.all(targets >= low - 1e-6) and np.all(targets <= high + 1e-6))
    return {"valid": bool(valid), "contact": contact, "max_penetration_m": float(pen),
            "finite": bool(finite), "joint_limit_violation": bool(joint_bad),
            "severe": bool(pen > PENETRATION_LIMIT),
            "quaternion_invalid": bool(abs(quat_norm - 1) > 1e-3),
            "action_in_range": bool(np.all(targets >= low - 1e-6) and np.all(targets <= high + 1e-6))}


def metric(x):
    s0, end = x[:, 0, :S_DIM], x[:, -1, :S_DIM]
    yaw = np.arctan2(end[:, 4], end[:, 3]) - np.arctan2(s0[:, 4], s0[:, 3])
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return {"dx_m": end[:, 0] - s0[:, 0], "dy_m": end[:, 1] - s0[:, 1],
            "yaw_deg": np.degrees(yaw), "height_m": end[:, 2] - s0[:, 2],
            "joint_rad": end[:, 14:26] - s0[:, 14:26]}


def wrap_deg(x):
    return (x + 180) % 360 - 180


def summarize(actual, pred):
    a, p = np.asarray(actual, np.float64).reshape(-1), np.asarray(pred, np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(p)
    a, p = a[finite], p[finite]
    if not len(a):
        return {"n": 0, "mjx_rms": None, "tt_rms": None, "mae": None, "rmse": None,
                "slope": None, "sign_agreement": None, "correlation": None}
    denom = float(a @ a)
    sign_mask = np.abs(a) > 1e-8
    corr = float(np.corrcoef(a, p)[0, 1]) if len(a) > 1 and np.std(a) > 1e-12 and np.std(p) > 1e-12 else None
    return {"n": len(a), "mjx_rms": float(np.sqrt(np.mean(a * a))), "tt_rms": float(np.sqrt(np.mean(p * p))),
            "mae": float(np.mean(np.abs(a - p))), "rmse": float(np.sqrt(np.mean((a - p) ** 2))),
            "slope": float(a @ p / denom) if denom > 1e-12 else None,
            "sign_agreement": float(np.mean(np.sign(a[sign_mask]) == np.sign(p[sign_mask]))) if sign_mask.any() else None,
            "correlation": corr}


def run_mjx(shape, qpos0, qvel0, target_q):
    env = envs[shape]
    count = len(qpos0)
    off, scale, (clip_low, clip_high) = np.asarray(env.action_offset), float(env.action_scale), env.clip_actions
    clipped = np.clip((target_q - off[:12]) / scale, clip_low, clip_high) * scale + off[:12]
    if not np.allclose(clipped, target_q, atol=1e-6, rtol=0):
        raise AssertionError(f"action clipping differs in {shape}")
    states = [make_data(env.mj_model, qpos=jnp.asarray(q), qvel=jnp.asarray(v), impl=env.impl,
                        nconmax=env._nconmax, njmax=env._njmax) for q, v in zip(qpos0, qvel0)]
    state = jax.tree.map(lambda *x: jnp.stack(x), *states)
    model = env.mjx_model
    state = jax.jit(jax.vmap(lambda s: mjx.forward(model, s)))(state)
    contacts0, initial_valid = [], []
    for i in range(count):
        q = np.asarray(state.qpos[i]).copy()
        v = np.asarray(state.qvel[i]).copy()
        d = mujoco.MjData(env.mj_model)
        d.qpos[:], d.qvel[:] = q, v
        mujoco.mj_forward(env.mj_model, d)
        c, pen = contact_and_penetration(env.mj_model, d, np.asarray(env._foot_ids), mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
        contacts0.append(c)
        initial_valid.append(np.isfinite(q).all() and np.isfinite(v).all() and pen <= PENETRATION_LIMIT)
    contacts0 = np.asarray(contacts0)
    vector_step = jax.jit(jax.vmap(lambda s, ctrl: step(model, s, ctrl, env.n_substeps)))
    q_hist, v_hist, c_hist = [], [], []
    fell = np.zeros(count, bool)
    fall_step = np.full(count, -1, np.int32)
    numerical = np.zeros(count, bool)
    for t in range(H):
        q, v = np.asarray(state.qpos), np.asarray(state.qvel)
        geom, dist = np.asarray(state._impl.contact.geom), np.asarray(state._impl.contact.dist)
        contacts = np.stack([((geom == foot).any(-1) & (dist <= 1e-3)).any(-1) for foot in np.asarray(env._foot_ids)], -1)
        up = 1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)
        finite = np.isfinite(q).all(1) & np.isfinite(v).all(1)
        fallen = (~finite) | (up < env.termination_fellover_threshold)
        new = fallen & ~fell
        fall_step[new] = t
        fell |= fallen
        numerical |= ~finite
        q_hist.append(q.copy()); v_hist.append(v.copy()); c_hist.append(contacts)
        if t == H - 1:
            break
        ctrl = np.broadcast_to(off, (count, len(off))).copy()
        ctrl[:, :12] = clipped[:, t]
        nxt = vector_step(state, jnp.asarray(ctrl))
        finite_nxt = np.isfinite(np.asarray(nxt.qpos)).all(1) & np.isfinite(np.asarray(nxt.qvel)).all(1)
        finite_nxt = jnp.asarray(finite_nxt)
        state = jax.tree.map(lambda a, b: jnp.where(finite_nxt.reshape((count,) + (1,) * (a.ndim - 1)), a, b), nxt, state)
    q_hist, v_hist, c_hist = np.stack(q_hist, 1), np.stack(v_hist, 1), np.stack(c_hist, 1)
    x = np.stack([np.concatenate((features(q_hist[i], v_hist[i], c_hist[i]), target_q[i]), axis=1) for i in range(count)])
    return {"qpos": q_hist, "qvel": v_hist, "contact": c_hist, "x": x,
            "initial_contact": contacts0, "initial_valid": np.asarray(initial_valid),
            "survived": ~fell, "fall_step": fall_step, "numerical_failure": numerical}


@torch.no_grad()
def predict(shape, x, pair_key_seed):
    count = len(x)
    x0 = (torch.as_tensor(x, device=device) - mu) / sd
    E = torch.as_tensor(data[shape]["E"], device=device)
    e = ((E - E_MU) / E_SD).expand(count, E_DIM)
    known, known_e = condition_masks(torch.ones(count, dtype=torch.long, device=device),
                                    torch.ones(count, H, dtype=torch.bool, device=device))
    pad = torch.zeros(count, H, dtype=torch.bool, device=device)
    samples = []
    for seed in range(args.tt_seeds):
        noise = torch.randn(x0.shape, generator=torch.Generator(device).manual_seed(args.seed + seed + pair_key_seed), device=device)
        xt = noise
        for k in reversed(range(len(beta))):
            ab, prev = abar[k], (abar[k - 1] if k else torch.ones((), device=device))
            xt = torch.where(known > 0, x0, xt)
            eps, _ = net(xt, e, known, known_e, torch.full((count,), k, device=device, dtype=torch.float32), pad)
            x0hat = ((xt - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-config["clip"], config["clip"])
            xt = prev.sqrt() * x0hat + (1 - prev).sqrt() * (xt - ab.sqrt() * x0hat) / (1 - ab).sqrt()
        samples.append((torch.where(known > 0, x0, xt) * sd + mu).cpu().numpy())
    return np.mean(samples, axis=0), np.stack(samples)


results: dict[str, Any] = {}
all_artifacts = {}
for pair_index, (shape_a, shape_b) in enumerate(PAIRS):
    pair_key = f"{shape_a}_{shape_b}"
    print(f"\n=== {shape_a} vs {shape_b} ===", flush=True)
    ranges = {}
    for shape in (shape_a, shape_b):
        env = envs[shape]
        off, scale = np.asarray(env.action_offset)[:12], float(env.action_scale)
        low, high = env.clip_actions
        ranges[shape] = (off + low * scale, off + high * scale)
    common_low = np.maximum(ranges[shape_a][0], ranges[shape_b][0])
    common_high = np.minimum(ranges[shape_a][1], ranges[shape_b][1])
    if np.any(common_low > common_high):
        raise ValueError(f"no common action range for {pair_key}")

    # Independent candidate pool and validity screening for this pair only.
    source_candidates = {}
    for source in (shape_a, shape_b):
        d = data[source]
        options = []
        for ep in test_episodes[source]:
            for t in range(0, d["valid"].shape[1] - H + 1, 8):
                if d["valid"][ep, t:t + H].all():
                    options.append((ep, t))
        rng.shuffle(options)
        options = options[:args.starts_per_source * args.pool_factor]
        kept, reason = [], defaultdict(int)
        for ep, t in options:
            q0 = d["qpos"][ep, t].astype(np.float64).copy()
            v0 = d["qvel"][ep, t].astype(np.float64).copy()
            action = d["target_q"][ep, t:t + H].astype(np.float32).copy()
            audits = {s: audit(s, q0, v0, action, common_low, common_high) for s in (shape_a, shape_b)}
            for s, ar in audits.items():
                for key in ("valid", "joint_limit_violation", "severe", "action_in_range"):
                    reason[f"{s}:{key}"] += int(ar.get(key, False))
            if all(audits[s]["valid"] for s in (shape_a, shape_b)) and np.array_equal(audits[shape_a]["contact"], audits[shape_b]["contact"]):
                kept.append({"source": source, "episode": int(ep), "t": int(t), "qpos0": q0, "qvel0": v0,
                             "target_q": action, "contact": audits[shape_a]["contact"]})
        # Round-robin episodes to prevent a few long survivors dominating.
        by_episode = defaultdict(list)
        rng.shuffle(kept)
        for row in kept:
            by_episode[row["episode"]].append(row)
        episodes = list(by_episode)
        rng.shuffle(episodes)
        chosen = []
        while episodes and len(chosen) < args.starts_per_source:
            remain = []
            for ep in episodes:
                if by_episode[ep] and len(chosen) < args.starts_per_source:
                    chosen.append(by_episode[ep].pop())
                if by_episode[ep]:
                    remain.append(ep)
            episodes = remain
        source_candidates[source] = {"rows": chosen, "audited": len(options), "valid_contact_matched": len(kept),
                                     "selected": len(chosen), "rejections": dict(reason)}
        print(f"{source}: audited={len(options)} valid+contact-matched={len(kept)} selected={len(chosen)}", flush=True)

    starts = source_candidates[shape_a]["rows"] + source_candidates[shape_b]["rows"]
    if len(starts) < 8:
        results[pair_key] = {"status": "insufficient pairwise-valid contact-matched starts", "selection": {
            s: {k: v for k, v in source_candidates[s].items() if k != "rows"} for s in (shape_a, shape_b)}}
        print(f"skip {pair_key}: only {len(starts)} matched starts", flush=True)
        continue
    qpos0 = np.stack([r["qpos0"] for r in starts])
    qvel0 = np.stack([r["qvel0"] for r in starts])
    target_q = np.stack([r["target_q"] for r in starts])
    source_labels = np.asarray([r["source"] for r in starts])
    episodes = np.asarray([r["episode"] for r in starts], np.int32)
    source_t = np.asarray([r["t"] for r in starts], np.int32)
    count = len(starts)
    mjx_runs = {shape_a: run_mjx(shape_a, qpos0, qvel0, target_q), shape_b: run_mjx(shape_b, qpos0, qvel0, target_q)}
    # Verify actual MJX initial contacts and exclude any CPU/MJX contact disagreement.
    contact_match = np.all(mjx_runs[shape_a]["initial_contact"] == mjx_runs[shape_b]["initial_contact"], axis=1)
    initial_valid = mjx_runs[shape_a]["initial_valid"] & mjx_runs[shape_b]["initial_valid"]
    keep = contact_match & initial_valid
    if not keep.any():
        results[pair_key] = {"status": "insufficient MJX-verified contact-matched starts", "candidate_count": count,
                             "mjx_contact_matched": int(contact_match.sum()), "mjx_initial_valid": int(initial_valid.sum()),
                             "selection": {s: {k: v for k, v in source_candidates[s].items() if k != "rows"} for s in (shape_a, shape_b)}}
        print(f"skip {pair_key}: MJX verified matched={keep.sum()}/{count}", flush=True)
        continue
    qpos0, qvel0, target_q = qpos0[keep], qvel0[keep], target_q[keep]
    source_labels, episodes, source_t = source_labels[keep], episodes[keep], source_t[keep]
    for shape in (shape_a, shape_b):
        for key in ("qpos", "qvel", "contact", "x", "survived", "fall_step", "numerical_failure", "initial_contact", "initial_valid"):
            mjx_runs[shape][key] = mjx_runs[shape][key][keep]
    count = int(keep.sum())
    # Exact paired S0 contact token and same actions; both morphology-specific E are the only conditions changed.
    tt_means, tt_by_seed = {}, {}
    for shape in (shape_a, shape_b):
        tt_means[shape], tt_by_seed[shape] = predict(shape, mjx_runs[shape]["x"], pair_index * 1000)
    truth = {s: metric(mjx_runs[s]["x"]) for s in (shape_a, shape_b)}
    preds = {s: metric(tt_means[s]) for s in (shape_a, shape_b)}
    actual = {k: truth[shape_b][k] - truth[shape_a][k] for k in truth[shape_a]}
    predicted = {k: preds[shape_b][k] - preds[shape_a][k] for k in preds[shape_a]}
    actual["yaw_deg"], predicted["yaw_deg"] = wrap_deg(actual["yaw_deg"]), wrap_deg(predicted["yaw_deg"])
    summary = {k: summarize(actual[k], predicted[k]) for k in actual}
    by_source = {}
    for source in (shape_a, shape_b):
        mask = source_labels == source
        by_source[source] = {k: summarize(actual[k][mask], predicted[k][mask]) for k in actual}
    # Absolute state error for each morphology on these exact common starts.
    abs_error = {}
    for shape in (shape_a, shape_b):
        truth_x, pred_x = mjx_runs[shape]["x"], tt_means[shape]
        abs_error[shape] = {}
        for name, lo, hi in (("position_xy", 0, 2), ("height", 2, 3), ("orientation_sincos", 3, 5),
                             ("gravity", 5, 8), ("linear_velocity", 8, 11), ("angular_velocity", 11, 14),
                             ("joint_position", 14, 26), ("joint_velocity", 26, 38), ("contact", 38, 42)):
            err = pred_x[..., lo:hi] - truth_x[..., lo:hi]
            if name == "orientation_sincos":
                yt = np.arctan2(truth_x[..., 4], truth_x[..., 3])
                yp = np.arctan2(pred_x[..., 4], pred_x[..., 3])
                err = ((yp - yt + np.pi) % (2 * np.pi) - np.pi)[..., None]
            abs_error[shape][name] = {"mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2)))}
        abs_error[shape]["survival"] = {"survived": int(mjx_runs[shape]["survived"].sum()), "total": count,
                                        "falls": int((~mjx_runs[shape]["survived"]).sum()),
                                        "numerical_failures": int(mjx_runs[shape]["numerical_failure"].sum())}
    pair_dir = args.out / pair_key
    pair_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"source_shape": source_labels, "source_episode": episodes, "source_t": source_t,
              "qpos0": qpos0, "qvel0": qvel0, "target_q": target_q,
              "morphology": np.stack([data[s]["E"] for s in (shape_a, shape_b)]),
              "initial_contact": np.stack([mjx_runs[s]["initial_contact"] for s in (shape_a, shape_b)]),
              "contact_matched": np.ones(count, bool),
              "states_mjx": np.stack([mjx_runs[s]["x"][..., :S_DIM] for s in (shape_a, shape_b)]),
              "states_tt": np.stack([tt_means[s][..., :S_DIM] for s in (shape_a, shape_b)]),
              "states_tt_by_seed": np.stack([tt_by_seed[s][..., :S_DIM] for s in (shape_a, shape_b)]),
              "qpos_mjx": np.stack([mjx_runs[s]["qpos"] for s in (shape_a, shape_b)]),
              "qvel_mjx": np.stack([mjx_runs[s]["qvel"] for s in (shape_a, shape_b)]),
              "contact_mjx": np.stack([mjx_runs[s]["contact"] for s in (shape_a, shape_b)]),
              "survived": np.stack([mjx_runs[s]["survived"] for s in (shape_a, shape_b)]),
              "fall_step": np.stack([mjx_runs[s]["fall_step"] for s in (shape_a, shape_b)]),
              "source_selected_counts": np.array([np.sum(source_labels == s) for s in (shape_a, shape_b)])}
    np.savez_compressed(pair_dir / "data.npz", **arrays)
    pair_result = {"status": "complete", "pair": [shape_a, shape_b], "groups": count,
                   "source_counts": {s: int(np.sum(source_labels == s)) for s in (shape_a, shape_b)},
                   "selection": {s: {k: v for k, v in source_candidates[s].items() if k != "rows"} for s in (shape_a, shape_b)},
                   "mjx_contact_matched": int(contact_match.sum()), "mjx_verified_kept": count,
                   "initial_contact_matched_groups": count, "shared_qpos_qvel_action": True,
                   "states": H, "executed_controls": H - 1, "control_period_s": 0.02,
                   "tt_seeds": [args.seed + pair_index * 1000 + s for s in range(args.tt_seeds)],
                   "metrics": summary, "metrics_by_source_shape": by_source, "absolute_state_error": abs_error}
    pair_result["source_strata_warning"] = [s for s in (shape_a, shape_b) if np.sum(source_labels == s) < 8]
    results[pair_key] = pair_result

    # Figure: response scatter for scalar metrics and RMS magnitude for 12 joints.
    fig, axes = plt.subplots(1, 5, figsize=(17, 4), layout="constrained")
    for ax, name in zip(axes, ("dx_m", "dy_m", "yaw_deg", "height_m", "joint_rad")):
        a, p = actual[name], predicted[name]
        if name == "joint_rad":
            a = np.sqrt(np.mean(a ** 2, axis=-1)); p = np.sqrt(np.mean(p ** 2, axis=-1))
        ax.scatter(a, p, s=12, alpha=0.45, c=np.where(source_labels == shape_a, "#4477AA", "#EE6677"))
        lim = max(float(np.nanmax(np.abs(np.r_[a, p]))), 1e-6)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), title=name, xlabel="MJX Δ", ylabel="TT Δ")
    fig.suptitle(f"{shape_b} − {shape_a}; blue/red = source morphology")
    fig.savefig(pair_dir / "response_scatter.png", dpi=170)
    plt.close(fig)

    # Mean trajectory overlay, stratified by source morphology.
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
    time = np.arange(H) * 0.02
    present_sources = [s for s in (shape_a, shape_b) if np.any(source_labels == s)]
    for col, source in enumerate(present_sources):
        mask = source_labels == source
        for row, (title, idx) in enumerate((("forward x (m)", 0), ("yaw (deg)", None), ("height Δ (m)", 2))):
            ax = axes[row, col]
            for shape, style in ((shape_a, "-"), (shape_b, "-")):
                for kind, values, ls in (("MJX", mjx_runs[shape]["x"], style), ("TT", tt_means[shape], "--")):
                    if title.startswith("yaw"):
                        y = wrap_deg(np.degrees(np.arctan2(values[mask, :, 4], values[mask, :, 3])))
                    else:
                        y = values[mask, :, idx] - values[mask, :1, idx]
                    ax.plot(time, np.mean(y, axis=0), ls, lw=1.3,
                            label=f"{shape} {kind}" if row == 0 else None)
            ax.set_title(f"source {source}: {title}"); ax.set_xlabel("time (s)"); ax.grid(alpha=.25)
            if row == 0:
                ax.legend(fontsize=7)
    for col in range(len(present_sources), 2):
        for row in range(3):
            axes[row, col].set_axis_off()
            axes[row, col].text(0.5, 0.5, "no selected source starts", ha="center", va="center")
    fig.savefig(pair_dir / "state_trajectories.png", dpi=170)
    plt.close(fig)
    print(f"{pair_key}: groups={count} source={pair_result['source_counts']}", flush=True)

summary = {"experiment": "D4 pairwise morphology sensitivity", "checkpoint": str(checkpoint_path),
           "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
           "checkpoint_config": {"horizon": int(config["horizon"]), "modes": list(config["modes"]),
                                 "diff_steps": int(config["diff_steps"]), "sampler": str(config["sampler"]),
                                 "norm": str(config["norm"]), "clip": float(config["clip"])},
           "source_split": "test episodes only", "states": H, "executed_controls": H-1,
           "same_qpos_qvel_A_within_pair": True, "additional_training": False,
           "morphologies": {s: np.asarray(data[s]["E"]).astype(float).tolist() for s in SHAPES},
           "seed": args.seed, "tt_seeds_per_pair": args.tt_seeds, "pairs": results}
(args.out / "d4_pairwise.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))

# Concise pairwise scoreboard.
lines = ["# D4 pairwise morphology sensitivity", "", f"Frozen checkpoint: `{checkpoint_path}`", "",
         "Each pair was sampled and screened independently. qpos, qvel and target_q are identical within pair; both starts must be valid and their initial four-foot MJX contact masks must match. No additional training or state correction.", "",
         "| Pair | groups (source A/B) | ΔMAE x (m) | y (m) | yaw (deg) | height (m) | joints (rad) |", "|---|---:|---:|---:|---:|---:|---:|"]
for key, result in results.items():
    if result["status"] != "complete":
        lines.append(f"| {key.replace('_',' ↔ ')} | insufficient | — | — | — | — | — |")
        continue
    m = result["metrics"]
    counts = result["source_counts"]
    pair_shapes = result["pair"]
    lines.append(f"| {key.replace('_',' ↔ ')} | {result['groups']} ({counts[pair_shapes[0]]}/{counts[pair_shapes[1]]}) | " +
                 " | ".join(f"{m[n]['mae']:.4g}" for n in ("dx_m","dy_m","yaw_deg","height_m","joint_rad")) + " |")
lines += ["", "## Pair-level details", ""]
for key, result in results.items():
    if result["status"] != "complete":
        lines += [f"### {key.replace('_',' ↔ ')}", "", result["status"], ""]
        continue
    m = result["metrics"]
    counts = result["source_counts"]
    pair_shapes = result["pair"]
    lines += ["", f"### {key.replace('_',' ↔ ')}", "",
              f"Source counts: {counts}; initial contacts match on all {result['groups']} groups. Falls: " +
              ", ".join(f"{s} {result['absolute_state_error'][s]['survival']['falls']}/{result['groups']}" for s in result['pair']) + ".",
              "", "| Response | MJX RMS Δ | TT RMS Δ | slope | sign agreement | ΔMAE |", "|---|---:|---:|---:|---:|---:|"]
    for metric_name, metric_result in m.items():
        lines.append(f"| {metric_name} | {metric_result['mjx_rms']:.4g} | {metric_result['tt_rms']:.4g} | "
                     f"{metric_result['slope'] if metric_result['slope'] is not None else float('nan'):.4g} | "
                     f"{metric_result['sign_agreement'] if metric_result['sign_agreement'] is not None else float('nan'):.3f} | {metric_result['mae']:.4g} |")
    lines += ["", f"Figures: `{key}/response_scatter.png`, `{key}/state_trajectories.png`."]
    if any(result["selection"][s]["selected"] == 0 for s in pair_shapes):
        lines += ["", "**Source imbalance:** one source morphology contributed zero starts that passed the pairwise physical/contact screen. This pair is valid as a same-start comparison, but it is not balanced across source rollout distributions."]
lines += ["", "Machine-readable per-start trajectories, starts, actions, contacts, falls and forecasts are in each pair's `data.npz`; detailed absolute errors and sample screening are in `d4_pairwise.json`.", ""]
(args.out / "README.md").write_text("\n".join(lines))
print(f"D4 pairwise complete -> {args.out}", flush=True)
