"""Same e000 MJX start, different PD target trajectories; save paired TT rollouts.

Run from codesign/: python code/tt_dynamics/collect_action_pairs.py --out runs/pipeline_v1/action_pairs/e000.npz
Each group_id has orig/swap/offset variants. A window has 32 states and 31 executed controls.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT.parent / "transformer-transformer"), str(ROOT / "code/motion_tracking")]

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from omegaconf import OmegaConf

from pd_replay import build_model
from t2.mjx_env.base_env import make_data, step
from track_env import TrackMjxEnv

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, default=ROOT / "runs/pipeline_v1/rollouts/e000/seq_seed0.npz")
parser.add_argument("--design", choices=("e000", "e001", "e002"), default="e000")
parser.add_argument("--starts-from", type=Path, help="reuse exact starts and orig/offset commands from a prior paired NPZ")
parser.add_argument("--manifest", type=Path, help="episode split and within-split swap donors")
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--starts", type=int, default=192)
parser.add_argument("--horizon", type=int, default=32)
parser.add_argument("--offset", type=float, default=0.2, help="per-joint constant PD target offset, radians")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
if args.starts < 2 or args.horizon < 2 or args.offset <= 0:
    parser.error("starts >= 2, horizon >= 2, offset > 0 required")
if bool(args.starts_from) != bool(args.manifest):
    parser.error("--starts-from and --manifest must be given together")

rng = np.random.default_rng(args.seed)
if args.starts_from:
    manifest = json.loads(args.manifest.read_text())
    with np.load(args.starts_from) as z:
        if "design" in z and str(z["design"]) != args.design:
            raise ValueError("--design does not match prior paired data")
        ids = z["group_id"][::3]
        if len(ids) * 3 != len(z["group_id"]) or args.horizon != z["target_q"].shape[1]:
            raise ValueError("prior paired data must contain three variants per start at the requested horizon")
        if ids.tolist() != [row["group_id"] for row in manifest["starts"]]:
            raise ValueError("manifest and prior paired data use different starts")
        q0, v0 = z["qpos"][::3, 0].copy(), z["qvel"][::3, 0].copy()
        tq, offset_tq = z["target_q"][::3].copy(), z["target_q"][2::3].copy()
        E = z["E"].astype(np.float32)
        b, t = z["source_episode"][::3], z["source_t"][::3]
    args.starts = len(ids)
    splits = np.array([row["split"] for row in manifest["starts"]])
    donor = np.array([row["swap_donor_group_id"] for row in manifest["starts"]])
    lookup = {int(g): i for i, g in enumerate(ids)}
    perm = np.array([lookup[int(d)] for d in donor])
    if any(splits[i] != splits[perm[i]] or i == perm[i] for i in range(args.starts)):
        raise ValueError("swap donor crosses split or is the same start")
else:
    with np.load(args.source) as z:
        if "design" in z and str(z["design"]) != args.design:
            raise ValueError("--design does not match source rollout")
        valid = z["valid"]
        candidates = np.argwhere(np.array([valid[:, t:t + args.horizon].all(1) for t in range(valid.shape[1] - args.horizon + 1)]).T)
        if len(candidates) < args.starts:
            raise ValueError(f"only {len(candidates)} valid source windows for {args.starts} requested starts")
        b, t = candidates[rng.choice(len(candidates), args.starts, replace=False)].T
        q0, v0 = z["qpos"][b, t].copy(), z["qvel"][b, t].copy()
        tq = np.stack([z["target_q"][bi, ti:ti + args.horizon] for bi, ti in zip(b, t)]).astype(np.float32)
        E = z["E"].astype(np.float32)

expert = ROOT / "runs/pipeline_v1/experts/walkForwardSlow__r00"
with open(sorted(expert.glob("wandb/offline-run-*/files/config.pkl"))[-1], "rb") as f:
    cfg = pickle.load(f)
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
choice = cfg.get("choice")
ecfg["reference_dir"] = str(ROOT / "runs/pipeline_v1/retargeting" / args.design / choice)
ecfg["clips"] = ["walkForwardSlow__r00"]
env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                                   torque_limit=cfg.get("torque_limit"), uniforms=E), **ecfg)
feet = np.asarray(env._foot_ids)
offset, scale, (low, high) = np.asarray(env.action_offset), float(env.action_scale), env.clip_actions
lo, hi = offset[:12] + low * scale, offset[:12] + high * scale
if tq.shape[-1] != 12 or not np.isfinite(tq).all():
    raise ValueError("expected finite 12-joint PD targets")

# Swap only inside the sampled source shape; each variant keeps the exact same qpos/qvel.
if not args.starts_from:
    perm = np.roll(np.arange(args.starts), rng.integers(1, args.starts))
    offset_tq = tq + rng.choice([-1, 1], size=(args.starts, 1, 12)) * args.offset
variants = np.stack([tq, tq[perm], offset_tq], 1)
variants = np.clip(variants, lo, hi).astype(np.float32)
q0 = np.repeat(q0, 3, axis=0)
v0 = np.repeat(v0, 3, axis=0)
targets = variants.reshape(-1, args.horizon, 12)
B = len(targets)
data = jax.tree.map(lambda *xs: jnp.stack(xs), *[
    make_data(env.mj_model, qpos=jnp.asarray(q), qvel=jnp.asarray(v),
              impl=env.impl, nconmax=env._nconmax, njmax=env._njmax)
    for q, v in zip(q0, v0)
])
data = jax.jit(jax.vmap(lambda state: mjx.forward(env.mjx_model, state)))(data)
vstep = jax.jit(jax.vmap(lambda state, ctrl: step(env.mjx_model, state, ctrl, env.n_substeps)))
record = {k: [] for k in ("qpos", "qvel", "contact", "valid")}
alive = np.ones(B, bool)
for ti in range(args.horizon):
    q = np.asarray(data.qpos)
    v = np.asarray(data.qvel)
    geom, dist = np.asarray(data._impl.contact.geom), np.asarray(data._impl.contact.dist)
    contact = np.stack([((geom == foot).any(-1) & (dist < 1e-3)).any(-1) for foot in feet], -1)
    up = 1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2)
    alive &= np.isfinite(q).all(1) & np.isfinite(v).all(1) & (up >= env.termination_fellover_threshold)
    for key, value in (("qpos", q), ("qvel", v), ("contact", contact), ("valid", alive.copy())):
        record[key].append(value)
    if ti < args.horizon - 1:
        ctrl = np.broadcast_to(offset, (B, len(offset))).copy()
        ctrl[:, :12] = targets[:, ti]
        data = vstep(data, jnp.asarray(ctrl))

out = {k: np.stack(v, axis=1) for k, v in record.items()}
out.update(target_q=targets, E=E, design=args.design, group_id=np.repeat(ids if args.starts_from else np.arange(args.starts), 3),
           variant=np.tile(np.array(["orig", "swap", "offset"]), args.starts),
           source_episode=np.repeat(b, 3), source_t=np.repeat(t, 3), dt=env.dt, seed=args.seed)
if args.starts_from:
    out.update(split=np.repeat(splits, 3), swap_donor_group_id=np.repeat(donor, 3))
keep = np.repeat((out["valid"].sum(1).reshape(-1, 3) >= 2).all(1), 3)
for key in ("qpos", "qvel", "contact", "valid", "target_q", "group_id", "variant", "source_episode", "source_t", "split", "swap_donor_group_id"):
    if key not in out:
        continue
    out[key] = out[key][keep]
if len(out["group_id"]) < 6:
    raise ValueError("fewer than two complete counterfactual groups remain")
for key in ("qpos", "qvel", "contact", "target_q"):
    assert np.isfinite(out[key]).all(), f"nonfinite {key}"
assert np.array_equal(out["qpos"][0::3, 0], out["qpos"][1::3, 0])
assert np.array_equal(out["qvel"][0::3, 0], out["qvel"][2::3, 0])
assert np.array_equal(out["target_q"][0::3], tq[keep.reshape(-1, 3)[:, 0]])
args.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.out, **out)
print(f"{len(out['group_id']) // 3}/{args.starts} starts kept, full windows {out['valid'].all(1).sum()}, "
      f"changed targets {np.count_nonzero(np.any(out['target_q'][1::3] != out['target_q'][0::3], axis=(1, 2)))} "
      f"swap pairs -> {args.out}")
