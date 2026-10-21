"""Controlled D4 morphology-sensitivity evaluation for the trained multi-shape TT.

For each held-out source window, keep qpos0, qvel0 and target_q identical while
replaying it on e000/e001/e002. No start-state height or pose correction is made
unless --align-height (start_align: body z only, source lowest-foot clearance kept).
The three TT predictions use the same per-group diffusion noise for each seed.

Run from codesign/:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
  ../transformer-transformer/.venv/bin/python code/tt_dynamics/d4_morphology.py \
    --out runs/pipeline_v1/diag/D4_morphology
"""

from __future__ import annotations

import argparse
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
from start_align import align_height, foot_clearance
from t2.mjx_env.base_env import make_data, step
from track_env import TrackMjxEnv
from tt_model import Net, condition_masks, schedule

SHAPES = ("e000", "e001", "e002")
PAIRS = ((1, 0), (2, 0), (2, 1))
H = 32
FOOT_R = 0.02
PENETRATION_LIMIT = 0.005

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--ckpt", type=Path, default=ROOT / "runs/pipeline_v1/tt/h32_multishape_B")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--starts-per-source", type=int, default=192)
parser.add_argument("--pool-factor", type=int, default=4, help="initially inspect this multiple to allow filtering")
parser.add_argument("--tt-seeds", type=int, default=8)
parser.add_argument("--seed", type=int, default=20261020)
parser.add_argument("--align-height", action="store_true",
                    help="per-target-E body z shift keeping the source lowest-foot clearance")
parser.add_argument("--audit-only", action="store_true", help="write selection.json after the start audit and stop")
parser.add_argument("--manifests", type=Path, nargs=3, default=[
    ROOT / "runs/pipeline_v1/action_ablation/split_manifest.json",
    ROOT / "runs/pipeline_v1/action_ablation/e001/split_manifest.json",
    ROOT / "runs/pipeline_v1/action_ablation/e002/split_manifest.json",
])
args = parser.parse_args()
if args.starts_per_source < 1 or args.pool_factor < 1 or args.tt_seeds < 1:
    parser.error("starts-per-source, pool-factor and tt-seeds must be positive")
if len(args.manifests) != 3:
    parser.error("exactly three held-out episode manifests are required")
if args.out.exists() and any(args.out.iterdir()):
    raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.out}")
args.out.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(args.seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Fixed model B and its training-time configuration ----
checkpoint_path = args.ckpt / "checkpoint.pt"
if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
config = checkpoint["args"]
if config["horizon"] != H or config.get("pose", "abs") != "abs" or config.get("modes") != [1]:
    raise ValueError(f"expected trained mode-1, absolute-pose, horizon-{H} model B; got {config}")
net = Net(H, config["d_model"], config["layers"]).to(device).eval()
net.load_state_dict(checkpoint["net"])
net.requires_grad_(False)
mu = torch.as_tensor(checkpoint["mu"], device=device)
sd = torch.as_tensor(checkpoint["sd"], device=device)
if torch.any(sd == 0):
    raise ValueError("checkpoint contains zero normalization scales")
E_MU, E_SD = ((0.5, 0.5) if config["norm"] == "minmax" else (0.5, 0.25))
beta, abar = schedule(config["diff_steps"], device)

# ---- Held-out source windows; each source morphology contributes equally ----
data_by_shape: dict[str, Any] = {}
manifest_by_shape: dict[str, dict[str, str]] = {}
for shape, manifest_path in zip(SHAPES, args.manifests):
    manifest = json.loads(manifest_path.read_text())
    episodes = {int(k) for k, split in manifest["episodes"].items() if split == "test"}
    if not episodes:
        raise ValueError(f"no test source episodes in {manifest_path}")
    d = np.load(ROOT / f"runs/pipeline_v1/rollouts/{shape}/seq_seed0.npz")
    if int(d["target_q"].shape[2]) != 12 or float(d["dt"]) != 0.02:
        raise ValueError(f"unexpected rollout action shape/control period for {shape}")
    if d["E"].shape != (E_DIM,) or not np.isfinite(d["E"]).all():
        raise ValueError(f"invalid morphology vector for {shape}")
    data_by_shape[shape] = d
    manifest_by_shape[shape] = {"path": str(manifest_path), "test_episodes": sorted(episodes)}

# Expert environment settings are used only to build the same MJX model/controller.
expert = ROOT / "runs/pipeline_v1/experts/walkForwardSlow__r00"
configs = sorted(expert.glob("wandb/offline-run-*/files/config.pkl"))
if not configs:
    raise FileNotFoundError(f"no expert config.pkl under {expert}")
with configs[-1].open("rb") as f:
    expert_cfg = pickle.load(f)
env_cfg = OmegaConf.to_container(expert_cfg.eval_env, resolve=True)
env_cfg.pop("_target_", None)
choice = expert_cfg.get("choice")
if not choice:
    raise ValueError("expert config has no morphology topology choice")

envs: dict[str, TrackMjxEnv] = {}
for shape in SHAPES:
    E = np.asarray(data_by_shape[shape]["E"], np.float32)
    cfg = dict(env_cfg)
    cfg["reference_dir"] = str(ROOT / f"runs/pipeline_v1/retargeting/{shape}/{choice}")
    cfg["clips"] = ["walkForwardSlow__r00"]
    envs[shape] = TrackMjxEnv(
        mj_model_fn=lambda E=E: build_model(choice, kp=expert_cfg.get("kp"),
                                             tau_scale=expert_cfg.get("tau_scale"),
                                             torque_limit=expert_cfg.get("torque_limit"), uniforms=E),
        **cfg,
    )

# Confirm a shared PD target range. Candidate windows outside the shared range are
# rejected rather than silently applying morphology-dependent action clipping.
target_ranges = {}
for shape, env in envs.items():
    off = np.asarray(env.action_offset)[:12]
    scale = float(env.action_scale)
    low, high = env.clip_actions
    target_ranges[shape] = (off + low * scale, off + high * scale)
shared_low = np.max(np.stack([target_ranges[s][0] for s in SHAPES]), axis=0)
shared_high = np.min(np.stack([target_ranges[s][1] for s in SHAPES]), axis=0)
if np.any(shared_low > shared_high):
    raise ValueError("the three morphologies have no common PD target range")


def contact_at(m: mujoco.MjModel, d: mujoco.MjData, feet: np.ndarray, floor: int) -> tuple[np.ndarray, float]:
    touching = np.zeros(len(feet), bool)
    max_penetration = 0.0
    foot_lookup = {int(g): i for i, g in enumerate(feet)}
    for c in d.contact[: d.ncon]:
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            if other in foot_lookup and c.dist <= 1e-3:
                touching[foot_lookup[other]] = True
                continue  # intended foot-floor contact is valid
        # Any other pair (including non-foot-floor and robot self-contact) is
        # unexpected at initialization when penetration exceeds the threshold.
        max_penetration = max(max_penetration, max(0.0, -float(c.dist)))
    return touching, max_penetration


def initial_audit(shape: str, qpos: np.ndarray, qvel: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    """Check unchanged start/action on one morphology; never adjust qpos or qvel."""
    env = envs[shape]
    m, d = env.mj_model, mujoco.MjData(env.mj_model)
    feet = np.asarray(env._foot_ids)
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(targets).all())
    joint_bad = False
    for jid in range(m.njnt):
        adr = int(m.jnt_qposadr[jid])
        if m.jnt_limited[jid] and not (m.jnt_range[jid, 0] - 1e-3 <= qpos[adr] <= m.jnt_range[jid, 1] + 1e-3):
            joint_bad = True
    qnorm = float(np.linalg.norm(qpos[3:7]))
    quat_bad = not np.isfinite(qnorm) or abs(qnorm - 1.0) > 1e-3
    try:
        mujoco.mj_forward(m, d)
        contacts, penetration = contact_at(m, d, feet, floor)
        forward_finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
    except (mujoco.FatalError, ValueError):
        contacts, penetration, forward_finite = np.zeros(4, bool), float("inf"), False
    in_range = bool(np.all(targets >= shared_low - 1e-6) and np.all(targets <= shared_high + 1e-6))
    severe_penetration = penetration > PENETRATION_LIMIT
    return {
        "contact": contacts,
        "max_floor_penetration_m": float(penetration),
        "finite": finite and forward_finite,
        "joint_limit_violation": bool(joint_bad),
        "quaternion_invalid": bool(quat_bad),
        "severe_floor_penetration": bool(severe_penetration),
        "action_in_common_range": in_range,
        "valid": finite and forward_finite and not joint_bad and not quat_bad and not severe_penetration and in_range,
    }

# Build a modest randomized candidate pool from Test episodes, with non-overlapping
# stride-8 starts. Then prefer same initial four-foot contact across the three E's.
candidates: dict[str, list[dict[str, Any]]] = {}
for shape in SHAPES:
    d = data_by_shape[shape]
    valid = np.asarray(d["valid"], bool)
    test_eps = manifest_by_shape[shape]["test_episodes"]
    possible = []
    for ep in test_eps:
        if ep >= len(valid):
            raise ValueError(f"manifest episode {ep} is out of range for {shape}")
        for t in range(0, valid.shape[1] - H + 1, 8):
            if valid[ep, t : t + H].all() and t + H <= d["qpos"].shape[1]:
                possible.append((ep, t))
    if not possible:
        raise ValueError(f"no valid 32-state windows in held-out episodes for {shape}")
    rng.shuffle(possible)
    pool_n = min(len(possible), args.starts_per_source * args.pool_factor)
    candidates[shape] = [
        {"source_shape": shape, "source_episode": int(ep), "source_t": int(t),
         "qpos0": np.asarray(d["qpos"][ep, t], np.float64).copy(),
         "qvel0": np.asarray(d["qvel"][ep, t], np.float64).copy(),
         "target_q": np.asarray(d["target_q"][ep, t : t + H], np.float32).copy()}
        for ep, t in possible[:pool_n]
    ]
    src_m, src_feet = envs[shape].mj_model, np.asarray(envs[shape]._foot_ids)
    for row in candidates[shape]:
        if args.align_height:
            clearance = foot_clearance(src_m, row["qpos0"], src_feet)
            row["qpos0_by_shape"] = {s: align_height(envs[s].mj_model, row["qpos0"], np.asarray(envs[s]._foot_ids), clearance)
                                     for s in SHAPES}
            assert abs(row["qpos0_by_shape"][shape][2] - row["qpos0"][2]) < 1e-9, "alignment must be identity on the source E"
        else:
            row["qpos0_by_shape"] = {s: row["qpos0"] for s in SHAPES}
    print(f"{shape}: {len(possible)} possible held-out windows; auditing {pool_n}", flush=True)

# Audit every candidate on all three E's. Starts and targets are copied bit-for-bit.
audited: dict[str, list[dict[str, Any]]] = {}
selection_counts: dict[str, Any] = {}
for source_shape in SHAPES:
    valid_rows = []
    reason_counts = defaultdict(int)
    for row in candidates[source_shape]:
        audits = {shape: initial_audit(shape, row["qpos0_by_shape"][shape], row["qvel0"], row["target_q"]) for shape in SHAPES}
        row["initial_audit"] = audits
        for shape, result in audits.items():
            for key in ("finite", "joint_limit_violation", "quaternion_invalid", "severe_floor_penetration", "action_in_common_range"):
                reason_counts[f"{shape}:{key}"] += int(result[key])
        if all(audits[s]["valid"] for s in SHAPES):
            row["contact_matched"] = bool(all(np.array_equal(audits[s]["contact"], audits[SHAPES[0]]["contact"]) for s in SHAPES[1:]))
            valid_rows.append(row)
    # Contact-matched starts are the preferred stratum. Round-robin episodes so
    # no single held-out episode dominates the capped sample.
    chosen = []
    for matched in (True, False):
        by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
        stratum = [r for r in valid_rows if r["contact_matched"] is matched]
        rng.shuffle(stratum)
        for row in stratum:
            by_episode[row["source_episode"]].append(row)
        episodes = list(by_episode)
        rng.shuffle(episodes)
        while episodes and len(chosen) < args.starts_per_source:
            next_episodes = []
            for ep in episodes:
                if by_episode[ep] and len(chosen) < args.starts_per_source:
                    chosen.append(by_episode[ep].pop())
                if by_episode[ep]:
                    next_episodes.append(ep)
            episodes = next_episodes
        if len(chosen) >= args.starts_per_source:
            break
    audited[source_shape] = chosen
    selection_counts[source_shape] = {
        "candidate_pool": len(candidates[source_shape]),
        "physically_valid": len(valid_rows),
        "contact_matched_valid": sum(r["contact_matched"] for r in valid_rows),
        "selected": len(chosen),
        "selected_contact_matched": sum(r["contact_matched"] for r in chosen),
        "audit_counts": dict(reason_counts),
    }
    print(f"{source_shape}: valid={len(valid_rows)}, selected={len(chosen)}, "
          f"contact-matched={selection_counts[source_shape]['selected_contact_matched']}", flush=True)

if args.audit_only:
    (args.out / "selection.json").write_text(json.dumps(
        {"align_height": args.align_height, "selection": selection_counts,
         "body_z_shift_m": {src: {s: [float(r["qpos0_by_shape"][s][2] - r["qpos0"][2]) for r in audited[src]] for s in SHAPES}
                            for src in SHAPES}}, indent=2))
    sys.exit(0)

if sum(selection_counts[s]["selected"] for s in SHAPES) < 32:
    raise ValueError(f"fewer than 32 shared physically valid starts across source strata: {selection_counts}")

# Flatten in fixed source order. Every group shares the same qpos0/qvel0/target_q
# across e000/e001/e002 executions; source_shape identifies the held-out corpus.
starts = [r for source_shape in SHAPES for r in audited[source_shape]]
source_shape = np.array([r["source_shape"] for r in starts])
source_episode = np.array([r["source_episode"] for r in starts], np.int32)
source_t = np.array([r["source_t"] for r in starts], np.int32)
qpos0 = np.stack([r["qpos0"] for r in starts]).astype(np.float64)
qpos0_by_shape = np.stack([[r["qpos0_by_shape"][s] for s in SHAPES] for r in starts]).astype(np.float64)
qvel0 = np.stack([r["qvel0"] for r in starts]).astype(np.float64)
target_q = np.stack([r["target_q"] for r in starts]).astype(np.float32)
N = len(starts)
assert all(np.isfinite(x).all() for x in (qpos0, qvel0, target_q))

# ---- MJX paired rollouts, exact qpos/qvel, 31 executed controls ----
def run_mjx(shape: str) -> dict[str, np.ndarray]:
    env = envs[shape]
    model = env.mjx_model
    feet = np.asarray(env._foot_ids)
    offset = np.asarray(env.action_offset)
    scale = float(env.action_scale)
    low, high = env.clip_actions
    clipped = np.clip((target_q - offset[:12]) / scale, low, high) * scale + offset[:12]
    if not np.allclose(clipped, target_q, atol=1e-6, rtol=0):
        raise AssertionError("a candidate escaped the shared action range after audit")
    datas = [make_data(env.mj_model, qpos=jnp.asarray(q), qvel=jnp.asarray(v), impl=env.impl,
                       nconmax=env._nconmax, njmax=env._njmax) for q, v in zip(qpos0_by_shape[:, SHAPES.index(shape)], qvel0)]
    state = jax.tree.map(lambda *xs: jnp.stack(xs), *datas)
    state = jax.jit(jax.vmap(lambda x: mjx.forward(model, x)))(state)
    init_q, init_v = np.asarray(state.qpos), np.asarray(state.qvel)
    if not np.isfinite(init_q).all() or not np.isfinite(init_v).all():
        raise ValueError(f"MJX produced non-finite initial state for {shape}; initial audit should have rejected it")
    vector_step = jax.jit(jax.vmap(lambda x, ctrl: step(model, x, ctrl, env.n_substeps)))
    q_hist, v_hist, c_hist = [], [], []
    fell = np.zeros(N, bool)
    fall_step = np.full(N, -1, np.int32)
    numerical_failure = np.zeros(N, bool)
    for ti in range(H):
        q = np.asarray(state.qpos)
        v = np.asarray(state.qvel)
        geom = np.asarray(state._impl.contact.geom)
        dist = np.asarray(state._impl.contact.dist)
        contacts = np.stack([((geom == foot).any(-1) & (dist <= 1e-3)).any(-1) for foot in feet], -1)
        finite = np.isfinite(q).all(1) & np.isfinite(v).all(1)
        up = 1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)
        fallen_now = (~finite) | (up < env.termination_fellover_threshold)
        new_fall = fallen_now & ~fell
        fall_step[new_fall] = ti
        fell |= fallen_now
        numerical_failure |= ~finite
        q_hist.append(q.copy())
        v_hist.append(v.copy())
        c_hist.append(contacts)
        if ti == H - 1:
            break
        ctrl = np.broadcast_to(offset, (N, len(offset))).copy()
        ctrl[:, :12] = clipped[:, ti]
        next_state = vector_step(state, jnp.asarray(ctrl))
        # If an unstable post-fall state becomes non-finite, retain the last finite
        # state for storage and flag its numerical failure; finite fallen rollouts continue.
        finite_next = jnp.asarray(np.isfinite(np.asarray(next_state.qpos)).all(1) &
                                  np.isfinite(np.asarray(next_state.qvel)).all(1))
        state = jax.tree.map(lambda nxt, old: jnp.where(finite_next.reshape((N,) + (1,) * (nxt.ndim - 1)), nxt, old), next_state, state)
    q_hist, v_hist, c_hist = np.stack(q_hist, 1), np.stack(v_hist, 1), np.stack(c_hist, 1)
    x = np.empty((N, H, S_DIM + 12), np.float32)
    for i in range(N):
        x[i, :, :S_DIM] = features(q_hist[i], v_hist[i], c_hist[i])
        x[i, :, S_DIM:] = target_q[i]
    return {"qpos": q_hist, "qvel": v_hist, "contact": c_hist, "x": x,
            "survived": ~fell, "fall_step": fall_step, "numerical_failure": numerical_failure}

mjx_runs = {}
for shape in SHAPES:
    mjx_runs[shape] = run_mjx(shape)
    print(f"{shape}: survived={int(mjx_runs[shape]['survived'].sum())}/{N}, "
          f"numerical failures={int(mjx_runs[shape]['numerical_failure'].sum())}", flush=True)

mjx_initial_contact = np.stack([mjx_runs[shape]["contact"][:, 0] for shape in SHAPES], axis=1)
source_index = np.array([SHAPES.index(s) for s in source_shape], dtype=np.int32)
common_s0_contact = mjx_initial_contact[np.arange(N), source_index]

# ---- TT forecasts. Same noise tensor per group is re-created for every E ----
@torch.no_grad()
def predict_for_shape(shape: str) -> tuple[np.ndarray, np.ndarray]:
    x = mjx_runs[shape]["x"].copy()
    # Contact is part of the learned state. Hold its S0 token fixed across E by
    # using the source morphology contact; later contacts remain actual MJX state.
    # With --align-height each E keeps its own actual MJX s0 contact.
    if not args.align_height:
        x[:, 0, 38:42] = common_s0_contact
    x0 = (torch.as_tensor(x, device=device) - mu) / sd
    E = np.asarray(data_by_shape[shape]["E"], np.float32)
    e = ((torch.as_tensor(E, device=device) - E_MU) / E_SD).expand(N, E_DIM)
    known, known_e = condition_masks(torch.ones(N, dtype=torch.long, device=device),
                                     torch.ones(N, H, dtype=torch.bool, device=device))
    pad = torch.zeros(N, H, dtype=torch.bool, device=device)
    seed_predictions = []
    for kseed in range(args.tt_seeds):
        noise = torch.randn(x0.shape, generator=torch.Generator(device).manual_seed(args.seed + kseed), device=device)
        xt = noise
        for k in reversed(range(len(beta))):
            ab = abar[k]
            prev = abar[k - 1] if k else torch.ones((), device=device)
            xt = torch.where(known > 0, x0, xt)
            eps, _ = net(xt, e, known, known_e, torch.full((N,), k, device=device, dtype=torch.float32), pad)
            denoised = ((xt - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-config["clip"], config["clip"])
            xt = prev.sqrt() * denoised + (1 - prev).sqrt() * (xt - ab.sqrt() * denoised) / (1 - ab).sqrt()
        pred = (torch.where(known > 0, x0, xt) * sd + mu).cpu().numpy()
        seed_predictions.append(pred)
    return np.mean(seed_predictions, axis=0), np.stack(seed_predictions)

pred_mean, pred_seeds = {}, {}
for shape in SHAPES:
    pred_mean[shape], pred_seeds[shape] = predict_for_shape(shape)
    print(f"{shape}: TT forecast complete ({args.tt_seeds} shared-noise seeds)", flush=True)


def final_metrics(x: np.ndarray) -> dict[str, np.ndarray]:
    """Per-group end displacement / change relative to the common initial state."""
    s0, end = x[:, 0, :S_DIM], x[:, -1, :S_DIM]
    yaw = np.arctan2(end[:, 4], end[:, 3]) - np.arctan2(s0[:, 4], s0[:, 3])
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return {
        "dx_m": end[:, 0] - s0[:, 0],
        "dy_m": end[:, 1] - s0[:, 1],
        "yaw_deg": np.degrees(yaw),
        "height_m": end[:, 2] - s0[:, 2],
        "joint_rad": end[:, 14:26] - s0[:, 14:26],
    }


def wrap_deg(x: np.ndarray) -> np.ndarray:
    return (x + 180.0) % 360.0 - 180.0


def summarize_delta(actual: np.ndarray, predicted: np.ndarray, name: str) -> dict[str, Any]:
    a = np.asarray(actual, np.float64).reshape(-1)
    p = np.asarray(predicted, np.float64).reshape(-1)
    finite = np.isfinite(a) & np.isfinite(p)
    a, p = a[finite], p[finite]
    if not len(a):
        return {"n_values": 0, "mjx_delta_rms": None, "tt_delta_rms": None, "delta_mae": None,
                "delta_rmse": None, "slope": None, "correlation": None, "sign_agreement": None,
                "n_sign_values": 0, "small_physical_delta": True}
    diff = p - a
    denom = float(np.dot(a, a))
    thresholds = {"dx_m": 1e-4, "dy_m": 1e-4, "yaw_deg": 0.01, "height_m": 1e-4, "joint_rad": 1e-4}
    sign_valid = np.abs(a) >= thresholds[name]
    corr = float(np.corrcoef(a, p)[0, 1]) if len(a) > 1 and np.std(a) > 1e-12 and np.std(p) > 1e-12 else None
    return {
        "n_values": int(len(a)),
        "mjx_delta_rms": float(np.sqrt(np.mean(a * a))),
        "tt_delta_rms": float(np.sqrt(np.mean(p * p))),
        "delta_mae": float(np.mean(np.abs(diff))),
        "delta_rmse": float(np.sqrt(np.mean(diff * diff))),
        "slope": float(np.dot(a, p) / denom) if denom > 1e-12 else None,
        "correlation": corr,
        "sign_agreement": float(np.mean(np.sign(a[sign_valid]) == np.sign(p[sign_valid]))) if sign_valid.any() else None,
        "n_sign_values": int(sign_valid.sum()),
        "small_physical_delta": bool(np.sqrt(np.mean(a * a)) < thresholds[name]),
    }

mjx_metric = {shape: final_metrics(mjx_runs[shape]["x"]) for shape in SHAPES}
tt_metric = {shape: final_metrics(pred_mean[shape]) for shape in SHAPES}
source_masks = {shape: source_shape == shape for shape in SHAPES}
cpu_contact_matched = np.array([bool(r["contact_matched"]) for r in starts])
contact_matched = np.all(mjx_initial_contact == mjx_initial_contact[:, :1], axis=(1, 2))
subsets = {"all": np.ones(N, bool), "initial_contact_matched": contact_matched}

pair_results: dict[str, Any] = {}
for pair_i, pair_j in PAIRS:
    pair_key = f"{SHAPES[pair_i]}-{SHAPES[pair_j]}"
    pair_results[pair_key] = {}
    for subset_name, subset in subsets.items():
        result_metrics = {}
        for name in ("dx_m", "dy_m", "yaw_deg", "height_m", "joint_rad"):
            actual_delta = mjx_metric[SHAPES[pair_i]][name] - mjx_metric[SHAPES[pair_j]][name]
            pred_delta = tt_metric[SHAPES[pair_i]][name] - tt_metric[SHAPES[pair_j]][name]
            if name == "yaw_deg":
                actual_delta, pred_delta = wrap_deg(actual_delta), wrap_deg(pred_delta)
            result_metrics[name] = summarize_delta(actual_delta[subset], pred_delta[subset], name)
            result_metrics[name]["by_source_shape"] = {}
            for source in SHAPES:
                local = subset & source_masks[source]
                local_a, local_p = actual_delta[local], pred_delta[local]
                if name == "yaw_deg":
                    local_a, local_p = wrap_deg(local_a), wrap_deg(local_p)
                result_metrics[name]["by_source_shape"][source] = summarize_delta(local_a, local_p, name)
        pair_results[pair_key][subset_name] = result_metrics

# Per-morphology absolute state-prediction errors; falls remain included. Numerical
# failures are counted and only their non-finite values are omitted per metric.
absolute_errors = {}
for shape in SHAPES:
    truth, prediction = mjx_runs[shape]["x"], pred_mean[shape]
    absolute_errors[shape] = {}
    fields = {
        "x_m": (0, 1), "y_m": (1, 2), "height_m": (2, 3),
        "yaw_deg": (3, 5), "gravity": (5, 8), "local_linear_velocity": (8, 11),
        "angular_velocity": (11, 14), "joint_position_rad": (14, 26),
        "joint_velocity": (26, 38), "contact": (38, 42),
    }
    for field, (lo, hi) in fields.items():
        if field == "yaw_deg":
            yt = np.arctan2(truth[..., 4], truth[..., 3])
            yp = np.arctan2(prediction[..., 4], prediction[..., 3])
            error = np.abs((yp - yt + np.pi) % (2 * np.pi) - np.pi) * (180 / np.pi)
        else:
            error = np.abs(prediction[..., lo:hi] - truth[..., lo:hi])
        finite = np.isfinite(error)
        absolute_errors[shape][field] = {
            "mae": float(np.mean(error[finite])) if finite.any() else None,
            "rmse": float(np.sqrt(np.mean(error[finite] ** 2))) if finite.any() else None,
            "n_values": int(finite.sum()),
        }
    absolute_errors[shape]["survival"] = {
        "survived": int(mjx_runs[shape]["survived"].sum()), "total": N,
        "fall_step_counts": {str(i): int((mjx_runs[shape]["fall_step"] == i).sum()) for i in range(H)},
        "numerical_failure": int(mjx_runs[shape]["numerical_failure"].sum()),
    }

# Seed-wise uncertainty for pair metrics. This is descriptive across diffusion noise,
# not a confidence interval over independent robots or morphologies.
seed_pair_mae: dict[str, Any] = {}
for pair_i, pair_j in PAIRS:
    pair_key = f"{SHAPES[pair_i]}-{SHAPES[pair_j]}"
    seed_pair_mae[pair_key] = {}
    for name in ("dx_m", "dy_m", "yaw_deg", "height_m", "joint_rad"):
        per_seed = []
        for si in range(args.tt_seeds):
            mi, mj = final_metrics(pred_seeds[SHAPES[pair_i]][si]), final_metrics(pred_seeds[SHAPES[pair_j]][si])
            delta = mi[name] - mj[name]
            if name == "yaw_deg":
                delta = wrap_deg(delta)
            actual = mjx_metric[SHAPES[pair_i]][name] - mjx_metric[SHAPES[pair_j]][name]
            if name == "yaw_deg":
                actual = wrap_deg(actual)
            err = np.abs(delta - actual)
            if name == "yaw_deg":
                err = np.abs(wrap_deg(delta - actual))
            per_seed.append(float(np.nanmean(err)))
        seed_pair_mae[pair_key][name] = {"mean": float(np.mean(per_seed)), "std": float(np.std(per_seed)),
                                         "per_seed": per_seed}

summary = {
    "experiment": "D4 morphology sensitivity, fixed qpos/qvel/action",
    "checkpoint": str(checkpoint_path),
    "checkpoint_sha256": __import__("hashlib").sha256(checkpoint_path.read_bytes()).hexdigest(),
    "checkpoint_config": {"horizon": int(config["horizon"]), "modes": list(config["modes"]),
                           "diff_steps": int(config["diff_steps"]), "sampler": str(config["sampler"]),
                           "norm": str(config["norm"]), "clip": float(config["clip"])},
    "control_period_s": 0.02, "states_per_window": H, "executed_control_intervals": H - 1,
    "same_initial_qpos_qvel_and_target_q": True,
    "no_initial_pose_or_height_correction": not args.align_height,
    "height_alignment": "body z per E, source lowest-foot clearance kept" if args.align_height else None,
    "source_split": "test episodes only",
    "manifests": manifest_by_shape,
    "morphologies": {shape: data_by_shape[shape]["E"].astype(float).tolist() for shape in SHAPES},
    "target_common_range_rad": {"low": shared_low.tolist(), "high": shared_high.tolist()},
    "seed": args.seed, "tt_diffusion_seeds": [args.seed + i for i in range(args.tt_seeds)],
    "group_count": N, "group_count_by_source_shape": {s: int(source_masks[s].sum()) for s in SHAPES},
    "selection": selection_counts,
    "initial_contact_matched_groups": int(contact_matched.sum()),
    "initial_contact_mismatch_groups": int((~contact_matched).sum()),
    "cpu_contact_matched_groups": int(cpu_contact_matched.sum()),
    "tt_s0_contact_source": ("each E's own MJX s0 contact" if args.align_height
                             else "held-out source morphology; identical across E within group"),
    "metrics_definition": {
        "dx_m/dy_m": "window-frame end displacement relative to t=0",
        "yaw_deg": "wrapped final heading change relative to t=0",
        "height_m": "final body z minus initial body z",
        "joint_rad": "12 final joint-angle changes relative to t=0; flattened for aggregate scores",
        "pair_sign": "e001-e000, e002-e000, e002-e001; yaw wrapped to [-180,180)",
        "slope": "sum(MJX_delta * TT_delta) / sum(MJX_delta^2), through origin",
        "small_delta_thresholds": {"dx_m": 1e-4, "dy_m": 1e-4, "yaw_deg": 0.01,
                                   "height_m": 1e-4, "joint_rad": 1e-4},
    },
    "pairs": pair_results,
    "absolute_state_error": absolute_errors,
    "diffusion_seed_delta_mae": seed_pair_mae,
}
(args.out / "d4_morphology.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))

# Preserve all source starts, exact actions, physical trajectories, and TT forecasts.
np.savez_compressed(
    args.out / "d4_morphology.npz",
    source_shape=source_shape, source_episode=source_episode, source_t=source_t,
    group_id=np.arange(N, dtype=np.int32), qpos0=qpos0, qpos0_by_shape=qpos0_by_shape, qvel0=qvel0, target_q=target_q,
    morphology=np.stack([data_by_shape[s]["E"] for s in SHAPES]).astype(np.float32),
    initial_contact_cpu=np.stack([[r["initial_audit"][s]["contact"] for s in SHAPES] for r in starts]).astype(bool),
    initial_contact_mjx=mjx_initial_contact.astype(bool),
    common_tt_s0_contact=common_s0_contact.astype(bool),
    initial_valid=np.stack([[r["initial_audit"][s]["valid"] for s in SHAPES] for r in starts]).astype(bool),
    initial_penetration_m=np.stack([[r["initial_audit"][s]["max_floor_penetration_m"] for s in SHAPES] for r in starts]).astype(np.float32),
    contact_matched=contact_matched,
    states_mjx=np.stack([mjx_runs[s]["x"][..., :S_DIM] for s in SHAPES]),
    states_tt=np.stack([pred_mean[s][..., :S_DIM] for s in SHAPES]),
    states_tt_by_seed=np.stack([pred_seeds[s][..., :S_DIM] for s in SHAPES]),
    qpos_mjx=np.stack([mjx_runs[s]["qpos"] for s in SHAPES]),
    qvel_mjx=np.stack([mjx_runs[s]["qvel"] for s in SHAPES]),
    contact_mjx=np.stack([mjx_runs[s]["contact"] for s in SHAPES]),
    survived=np.stack([mjx_runs[s]["survived"] for s in SHAPES]),
    fall_step=np.stack([mjx_runs[s]["fall_step"] for s in SHAPES]),
    numerical_failure=np.stack([mjx_runs[s]["numerical_failure"] for s in SHAPES]),
)

# ---- Five requested figures ----
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25})
colors = {"e000": "#4477AA", "e001": "#EE6677", "e002": "#228833"}
metric_names = ("dx_m", "dy_m", "yaw_deg", "height_m", "joint_rad")
metric_labels = ("Δx (m)", "Δy (m)", "Δyaw (deg)", "Δheight (m)", "Δjoint RMS (rad)")

# 1) MJX and TT mean morphology-induced changes, for each source stratum.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False, layout="constrained")
for ax, source in zip(axes, SHAPES):
    local = source_masks[source]
    xloc = np.arange(len(metric_names))
    for pi, (i, j) in enumerate(PAIRS[:2]):
        delta_m = []
        delta_t = []
        for name in metric_names:
            am = mjx_metric[SHAPES[i]][name][local] - mjx_metric[SHAPES[j]][name][local]
            at = tt_metric[SHAPES[i]][name][local] - tt_metric[SHAPES[j]][name][local]
            if name == "yaw_deg":
                am, at = wrap_deg(am), wrap_deg(at)
            if not local.any():
                delta_m.append(float("nan"))
                delta_t.append(float("nan"))
                continue
            if name == "joint_rad":
                delta_m.append(float(np.nanmean(np.sqrt(np.nanmean(am ** 2, axis=-1)))))
                delta_t.append(float(np.nanmean(np.sqrt(np.nanmean(at ** 2, axis=-1)))))
            else:
                delta_m.append(float(np.nanmean(am)))
                delta_t.append(float(np.nanmean(at)))
        width = 0.23
        pos = xloc + (pi - 1) * width
        ax.bar(pos - width * 0.18, delta_m, width * 0.36, color=colors[SHAPES[i]], alpha=0.9,
               label=f"MJX {SHAPES[i]}-{SHAPES[j]}" if source == SHAPES[0] else None)
        ax.bar(pos + width * 0.18, delta_t, width * 0.36, color=colors[SHAPES[i]], alpha=0.42, hatch="//",
               label=f"TT {SHAPES[i]}-{SHAPES[j]}" if source == SHAPES[0] else None)
    ax.set_xticks(xloc, metric_labels, rotation=20, ha="right")
    ax.set_title(f"source {source} (n={local.sum()})")
    ax.axhline(0, color="black", linewidth=0.7)
axes[0].set_ylabel("Mean pairwise response (joint: RMS magnitude)")
fig.legend(loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.12))
fig.savefig(args.out / "figure1_morphology_changes.png", dpi=170)
plt.close(fig)

# 2) Pairwise ΔMAE by state dimension.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=False, layout="constrained")
for ax, source in zip(axes, SHAPES):
    local = source_masks[source]
    for pi, (i, j) in enumerate(PAIRS):
        vals = []
        for name in metric_names:
            a = mjx_metric[SHAPES[i]][name][local] - mjx_metric[SHAPES[j]][name][local]
            p = tt_metric[SHAPES[i]][name][local] - tt_metric[SHAPES[j]][name][local]
            if name == "yaw_deg":
                a, p = wrap_deg(a), wrap_deg(p)
            if not local.any():
                vals.append(float("nan"))
                continue
            err = np.abs(p - a)
            if name == "yaw_deg":
                err = np.abs(wrap_deg(p - a))
            vals.append(float(np.nanmean(err)))
        ax.plot(metric_labels, vals, marker="o", label=f"{SHAPES[i]}-{SHAPES[j]}")
    ax.set_title(f"source {source}")
    ax.tick_params(axis="x", rotation=24)
    ax.set_ylabel("ΔMAE")
    ax.legend()
fig.savefig(args.out / "figure2_delta_mae.png", dpi=170)
plt.close(fig)

# 3) Response slopes by state dimension, with ideal slope 1.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True, layout="constrained")
for ax, source in zip(axes, SHAPES):
    local = source_masks[source]
    for i, j in PAIRS:
        vals = []
        for name in metric_names:
            a = mjx_metric[SHAPES[i]][name][local] - mjx_metric[SHAPES[j]][name][local]
            p = tt_metric[SHAPES[i]][name][local] - tt_metric[SHAPES[j]][name][local]
            if name == "yaw_deg":
                a, p = wrap_deg(a), wrap_deg(p)
            if not local.any():
                vals.append(float("nan"))
                continue
            a, p = np.asarray(a).reshape(-1), np.asarray(p).reshape(-1)
            finite = np.isfinite(a) & np.isfinite(p)
            den = np.dot(a[finite], a[finite]) if finite.any() else 0.0
            vals.append(float(np.dot(a[finite], p[finite]) / den) if den > 1e-12 else np.nan)
        ax.plot(metric_labels, vals, marker="o", label=f"{SHAPES[i]}-{SHAPES[j]}")
    ax.axhline(1, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_title(f"source {source}")
    ax.tick_params(axis="x", rotation=24)
    ax.legend()
axes[0].set_ylabel("Through-origin response slope")
fig.savefig(args.out / "figure3_slope.png", dpi=170)
plt.close(fig)

# 4) Individual-start scatter, combined across held-out source strata.
fig, axes = plt.subplots(1, 5, figsize=(19, 4.1), layout="constrained")
scatter_metrics = ("dx_m", "dy_m", "yaw_deg", "height_m", "joint_rad")
for ax, name in zip(axes, scatter_metrics):
    all_values = []
    for i, j in PAIRS:
        actual = mjx_metric[SHAPES[i]][name] - mjx_metric[SHAPES[j]][name]
        pred = tt_metric[SHAPES[i]][name] - tt_metric[SHAPES[j]][name]
        if name == "yaw_deg":
            actual, pred = wrap_deg(actual), wrap_deg(pred)
        if name == "joint_rad":
            actual, pred = actual.reshape(-1), pred.reshape(-1)
        if name == "joint_rad":
            # Plot each start's joint RMS response magnitude while preserving
            # the signed per-joint values in the machine-readable artifact.
            actual = np.sqrt(np.mean(actual.reshape(-1, 12) ** 2, axis=1))
            pred = np.sqrt(np.mean(pred.reshape(-1, 12) ** 2, axis=1))
        ax.scatter(actual, pred, s=9, alpha=0.32, label=f"{SHAPES[i]}-{SHAPES[j]}")
        if np.isfinite(actual).any() and np.isfinite(pred).any():
            all_values.extend([actual[np.isfinite(actual)], pred[np.isfinite(pred)]])
    finite_values = np.concatenate(all_values) if all_values else np.array([-1, 1])
    if len(finite_values):
        lim = max(float(np.nanmax(np.abs(finite_values))), 1e-6)
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.set_title(name)
    ax.set_xlabel("MJX Δ")
    ax.set_ylabel("TT Δ")
axes[-1].legend(fontsize=7)
fig.savefig(args.out / "figure4_pair_scatter.png", dpi=170)
plt.close(fig)

# 5) Mean physical-state trajectories by source stratum, morphology, and simulator.
fig, axes = plt.subplots(3, 3, figsize=(15, 10), layout="constrained")
time = np.arange(H) * 0.02
for col, source in enumerate(SHAPES):
    local = source_masks[source]
    for row, (name, index, unit) in enumerate((("x", 0, "m"), ("yaw", None, "deg"), ("height", 2, "m"))):
        ax = axes[row, col]
        # Compare the reference morphology with the two interventions requested
        # in the primary D4 question; e002-vs-e001 remains a summary-only pair.
        for shape in SHAPES:
            for kind, series, style in (("MJX", mjx_runs[shape]["x"], "-"), ("TT", pred_mean[shape], "--")):
                if name == "yaw":
                    y = np.degrees(np.arctan2(series[local, :, 4], series[local, :, 3]))
                    y = wrap_deg(y)
                elif name == "height":
                    y = series[local, :, index] - series[local, :1, index]
                else:
                    y = series[local, :, index] - series[local, :1, index]
                with np.errstate(invalid="ignore"):
                    mean = np.nanmean(y, axis=0)
                ax.plot(time, mean, style, color=colors[shape], linewidth=1.4,
                        label=f"{shape} {kind}" if col == 0 else None)
        ax.set_title(f"source {source}: {name}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(unit)
axes[0, 0].legend(fontsize=7, ncol=2)
fig.savefig(args.out / "figure5_state_trajectories.png", dpi=170)
plt.close(fig)

# A compact Markdown result report is convenient for the research handoff.
lines = ["# D4 morphology sensitivity results", "", f"Checkpoint: `{args.ckpt}`",
         f"Starts: {N} total ({', '.join(f'{s}={int(source_masks[s].sum())}' for s in SHAPES)} source starts); "
         f"initial-contact matched: {int(contact_matched.sum())}; TT diffusion seeds: {args.tt_seeds}.",
         "", ("The same qpos0 (body z shifted per E to keep the source lowest-foot clearance), qvel0 and target_q"
              if args.align_height else "The same unmodified qpos0, qvel0 and target_q") + " are replayed on each morphology."
         " Each window has 32 states and 31 executed 50 Hz controls. Falls remain included;"
         " numerical failures are flagged and only non-finite metric entries are omitted.",
         "", "## Physical-validity selection", "",
         "| Source morphology | Audited candidates | Valid across all E | Selected | Contact-matched selected |",
         "|---|---:|---:|---:|---:|"]
for source in SHAPES:
    selection = selection_counts[source]
    lines.append(f"| {source} | {selection['candidate_pool']} | {selection['physically_valid']} | "
                 f"{selection['selected']} | {selection['selected_contact_matched']} |")
lines += ["", "The held-out source strata are strongly imbalanced because the same unmodified start must avoid >5 mm non-foot floor penetration on all three bodies. "
          "The aggregate is therefore dominated by e001-origin windows and is not a balanced three-stratum estimate. "
          "Only 13 groups have matching initial contacts in all three MJX models; those are reported separately in JSON.",
          "", "## Pairwise response ΔMAE (all valid groups)", "", "| Pair | dx (m) | dy (m) | yaw (deg) | height (m) | joints (rad) |", "|---|---:|---:|---:|---:|---:|"]
for pair, subsets_result in pair_results.items():
    m = subsets_result["all"]
    lines.append(f"| {pair} | " + " | ".join(f"{m[k]['delta_mae']:.4g}" if m[k]["delta_mae"] is not None else "n/a"
                                                  for k in metric_names) + " |")
lines += ["", "## Figures", "", "- figure1_morphology_changes.png: MJX vs TT morphology-induced response.",
          "- figure2_delta_mae.png: response ΔMAE by source morphology.",
          "- figure3_slope.png: through-origin slopes; ideal = 1.",
          "- figure4_pair_scatter.png: per-start MJX vs TT response.",
          "- figure5_state_trajectories.png: mean state trajectories by source morphology.",
          "", "See d4_morphology.json for per-source, contact-matched, fall, absolute-error and diffusion-seed details;",
          "see d4_morphology.npz for per-group starts, trajectories and predictions.", ""]
(args.out / "README.md").write_text("\n".join(lines))
print(f"D4 done: {N} groups -> {args.out}", flush=True)
