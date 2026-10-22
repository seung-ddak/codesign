"""Collect action and strict morphology interventions from foot-anchored starts."""
import argparse, gc, glob, hashlib, json, pickle, sys
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

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--starts", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
p.add_argument("--actions", nargs="+", default=glob.glob("runs/pipeline_v1/wide16_action/rollouts/e*/primitive_seed0.npz"))
p.add_argument("--expert-config", type=Path, default=ROOT / "runs/pipeline_v1/experts_wide16/walkForwardSlow__r00")
p.add_argument("--horizon", type=int, default=32); p.add_argument("--offset", type=float, default=.2)
p.add_argument("--min-action-rms", type=float, default=.05); p.add_argument("--seed", type=int, default=0)
p.add_argument("--command-bank-max", type=int, default=8192)
p.add_argument("--rollout-batch", type=int, default=3); args = p.parse_args()
if args.horizon < 2 or args.offset <= 0 or args.min_action_rms <= 0 or args.command_bank_max < 2:
    p.error("invalid horizon, offset, action RMS, or command bank size")
if args.rollout_batch < 1:
    p.error("rollout-batch must be positive")
args.out.mkdir(parents=True, exist_ok=True); done = args.out / "prepare.done"; done.unlink(missing_ok=True)

configs = sorted(args.expert_config.glob("wandb/offline-run-*/files/config.pkl"))
if not configs: raise FileNotFoundError(f"No expert config under {args.expert_config}")
with configs[-1].open("rb") as f: cfg = pickle.load(f)
choice = cfg.get("choice"); ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True); ecfg.pop("_target_", None)
ecfg.update(reference_dir=str(ROOT / "runs/pipeline_v1/retargeting/e000" / choice), clips=["walkForwardSlow__r00"])
rng = np.random.default_rng(args.seed)

bank_q, bank_a = [], []
per_source = max(1, args.command_bank_max // len(args.actions))
for path in args.actions:
    with np.load(path) as z:
        candidates = [(b, t) for b in range(len(z["valid"])) for t in range(0, z["valid"].shape[1] - args.horizon + 1, args.horizon)
                      if z["valid"][b, t:t + args.horizon].all()]
        if len(candidates) > per_source:
            candidates = [candidates[i] for i in rng.choice(len(candidates), per_source, replace=False)]
        bank_q.extend(z["qpos"][b, t, 7:19] for b, t in candidates)
        bank_a.extend(z["target_q"][b, t:t + args.horizon] for b, t in candidates)
if len(bank_a) < 2: raise ValueError("fewer than two valid expert command windows")
bank_q, bank_a = np.asarray(bank_q), np.asarray(bank_a, np.float32)
with np.load(args.starts) as z: starts = {k: z[k] for k in z.files}
for key in ("E", "design_id", "split", "qpos", "qvel"):
    if key not in starts: raise KeyError(f"{key} missing in {args.starts}")
designs = np.unique(starts["design_id"])
if len(designs) < 3: raise ValueError(f"at least three morphologies are required, got {len(designs)}")

def make_env(E):
    return TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                                       torque_limit=cfg.get("torque_limit"), uniforms=E), **ecfg)

E_by_design = {int(d): starts["E"][np.flatnonzero(starts["design_id"] == d)[0]].astype(np.float32) for d in designs}

def initial_valid(env, q, v):
    m, d = env.mj_model, mujoco.MjData(env.mj_model); d.qpos[:], d.qvel[:] = q, v
    if not np.isfinite(q).all() or not np.isfinite(v).all() or abs(np.linalg.norm(q[3:7]) - 1) > 1e-3: return False
    for jid in range(m.njnt):
        adr = int(m.jnt_qposadr[jid])
        if m.jnt_limited[jid] and not (m.jnt_range[jid, 0] - 1e-6 <= q[adr] <= m.jnt_range[jid, 1] + 1e-6): return False
    mujoco.mj_forward(m, d); feet = set(map(int, env._foot_ids)); floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"); touching = set()
    for c in d.contact[:d.ncon]:
        if floor in (c.geom1, c.geom2):
            other = int(c.geom2 if c.geom1 == floor else c.geom1)
            if other in feet and c.dist <= 1e-3: touching.add(other)
            elif c.dist < -.002: return False
        elif c.dist < -.002: return False
    return len(touching) == 4

def rollout(env, q0, v0, targets):
    B = len(q0); model, feet = env.mj_model, np.asarray(env._foot_ids)
    data = jax.tree.map(lambda *xs: jnp.stack(xs), *[make_data(model, qpos=jnp.asarray(q), qvel=jnp.asarray(v), impl=env.impl,
        nconmax=env._nconmax, njmax=env._njmax) for q, v in zip(q0, v0)])
    data = jax.jit(jax.vmap(lambda x: mjx.forward(env.mjx_model, x)))(data)
    vstep = jax.jit(jax.vmap(lambda x, u: step(env.mjx_model, x, u, env.n_substeps)))
    record = {k: [] for k in ("qpos", "qvel", "contact", "valid")}; alive = np.ones(B, bool); initial_ok = np.zeros(B, bool)
    for t in range(args.horizon):
        q, v = np.asarray(data.qpos), np.asarray(data.qvel); geom, dist = np.asarray(data._impl.contact.geom), np.asarray(data._impl.contact.dist)
        touching = np.stack([((geom == foot).any(-1) & (dist <= 1e-3)).any(-1) for foot in feet], -1)
        if t == 0: initial_ok = touching.all(1) & np.isfinite(q).all(1) & np.isfinite(v).all(1)
        alive &= np.isfinite(q).all(1) & np.isfinite(v).all(1) & (1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2) >= env.termination_fellover_threshold)
        for k, x in (("qpos", q), ("qvel", v), ("contact", touching), ("valid", alive.copy())): record[k].append(x)
        if t < args.horizon - 1:
            ctrl = np.broadcast_to(np.asarray(env.action_offset), (B, env.mj_model.nu)).copy(); ctrl[:, :12] = targets[:, t]
            data = vstep(data, jnp.asarray(ctrl))
    return {k: np.stack(x, 1) for k, x in record.items()}, initial_ok

def rollout_chunks(env, q0, v0, targets):
    """Keep the JAX device buffer bounded while retaining the same trajectories."""
    records, valid = [], []
    for lo in range(0, len(q0), args.rollout_batch):
        rec, ok = rollout(env, q0[lo:lo + args.rollout_batch], v0[lo:lo + args.rollout_batch],
                          targets[lo:lo + args.rollout_batch])
        records.append(rec); valid.append(ok)
    return {k: np.concatenate([r[k] for r in records]) for k in records[0]}, np.concatenate(valid)

def save(path, out):
    tmp = path.with_suffix(".tmp.npz"); np.savez_compressed(tmp, **out); tmp.replace(path)

base = {}
for design_id in map(int, designs):
    idx = np.flatnonzero(starts["design_id"] == design_id); E, env = E_by_design[design_id], make_env(E_by_design[design_id])
    off, scale, (low, high) = np.asarray(env.action_offset), float(env.action_scale), env.clip_actions
    lo, hi = off[:12] + low * scale, off[:12] + high * scale; q0, v0 = starts["qpos"][idx], starts["qvel"][idx]
    nearest = np.argmin(((q0[:, None, 7:19] - bank_q[None]) ** 2).mean(2), 1); orig = bank_a[nearest]; donor = np.empty(len(idx), int)
    for split in np.unique(starts["split"][idx]):
        local = np.flatnonzero(starts["split"][idx] == split)
        for i in local:
            rms = np.sqrt(np.mean((orig[local] - orig[i]) ** 2, axis=(1, 2))); donor[i] = local[np.argmax(rms)]
    donor_source_idx = idx[donor]
    signs = rng.choice([-1., 1.], (len(idx), 1, 12)); variants = np.clip(np.stack([orig, orig[donor], orig + signs * args.offset], 1), lo, hi).astype(np.float32)
    delta = np.sqrt(np.mean((variants[:, 1:] - variants[:, :1]) ** 2, axis=(2, 3)))
    keep = (delta >= args.min_action_rms).all(1) & np.array([initial_valid(env, q, v) for q, v in zip(q0, v0)])
    idx, q0, v0, variants, donor_source_idx, delta = idx[keep], q0[keep], v0[keep], variants[keep], donor_source_idx[keep], delta[keep]
    if not len(idx): raise RuntimeError(f"design {design_id}: no valid distinct-action starts")
    B = len(idx) * 3; targets = variants.reshape(B, args.horizon, 12)
    rec, mjx_ok = rollout_chunks(env, np.repeat(q0, 3, 0), np.repeat(v0, 3, 0), targets); group_ok = mjx_ok.reshape(-1, 3).all(1); rows = np.repeat(group_ok, 3)
    group = design_id * 1_000_000 + idx; out = {k: x[rows] for k, x in rec.items()}
    out.update(target_q=targets[rows], E=E, design=f"f{design_id:03d}", design_id=np.full(rows.sum(), design_id),
        source_design_id=np.full(rows.sum(), design_id), group_id=np.repeat(group, 3)[rows], action_group_id=np.repeat(group, 3)[rows],
        morph_group_id=np.full(rows.sum(), -1), variant=np.tile(["orig", "swap", "offset"], len(idx))[rows],
        intervention_type=np.tile(["orig", "swap", "offset"], len(idx))[rows], split=np.repeat(starts["split"][idx], 3)[rows],
        source_episode=np.repeat(idx, 3)[rows], source_t=np.zeros(rows.sum(), np.int32),
        swap_donor_group_id=np.repeat(design_id * 1_000_000 + donor_source_idx, 3)[rows],
        action_delta_rms=np.c_[np.zeros(len(idx)), delta].reshape(-1)[rows], initial_valid=np.ones(rows.sum(), bool), dt=.02, seed=args.seed)
    path = args.out / f"f{design_id:03d}.npz"; save(path, out); orig_rows = out["variant"] == "orig"
    save(args.out / f"f{design_id:03d}_orig.npz", {k: (v[orig_rows] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(rows[rows]) else v) for k, v in out.items()})
    kept = idx[group_ok]; manifest = {"seed": args.seed, "source": str(path), "episodes": {str(int(i)): str(starts["split"][i]) for i in kept},
        "starts": [{"group_id": int(design_id * 1_000_000 + i), "source_episode": int(i), "source_t": 0, "split": str(starts["split"][i])} for i in kept]}
    (args.out / f"f{design_id:03d}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    base[design_id] = dict(idx=kept, q=q0[group_ok], v=v0[group_ok], action=variants[group_ok, 0], group=group[group_ok])
    print(f"{path}: {len(kept)} starts, min action RMS {delta[group_ok].min():.3f}", flush=True)
    del env
    jax.clear_caches(); gc.collect()

morph_count = 0
for source in map(int, designs):
    target = min((int(d) for d in designs if d != source), key=lambda d: np.linalg.norm(E_by_design[source] - E_by_design[d])); b = base[source]
    target_env = make_env(E_by_design[target])
    cpu = np.array([initial_valid(target_env, q, v) for q, v in zip(b["q"], b["v"])])
    if not cpu.any():
        print(f"f{source:03d}->f{target:03d}: no shared starts", flush=True)
        del target_env
        jax.clear_caches(); gc.collect()
        continue
    rec, mjx_ok = rollout_chunks(target_env, b["q"][cpu], b["v"][cpu], b["action"][cpu]); source_idx = b["idx"][cpu][mjx_ok]; group = b["group"][cpu][mjx_ok]
    if not len(group): continue
    out = {k: x[mjx_ok] for k, x in rec.items()}; n = len(group)
    out.update(target_q=b["action"][cpu][mjx_ok], E=E_by_design[target], design=f"f{target:03d}", design_id=np.full(n, target),
        source_design_id=np.full(n, source), group_id=group, action_group_id=group, morph_group_id=group,
        variant=np.full(n, "morphology"), intervention_type=np.full(n, "morphology"), split=starts["split"][source_idx],
        source_episode=source_idx, source_t=np.zeros(n, np.int32), swap_donor_group_id=np.full(n, -1), action_delta_rms=np.zeros(n),
        initial_valid=np.ones(n, bool), dt=.02, seed=args.seed)
    save(args.out / f"morph_f{source:03d}_to_f{target:03d}.npz", out); morph_count += n
    print(f"strict morphology f{source:03d}->f{target:03d}: {n} starts", flush=True)
    del target_env
    jax.clear_caches(); gc.collect()
if not morph_count: raise RuntimeError("no strict morphology intervention survived")
summary = {"designs": len(designs), "morphology_rollouts": morph_count, "horizon": args.horizon, "min_action_rms": args.min_action_rms,
           "starts_sha256": hashlib.sha256(args.starts.read_bytes()).hexdigest()}
done.write_text(json.dumps(summary, indent=2) + "\n"); print("complete:", summary)
