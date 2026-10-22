"""Generate physically checked four-foot initial states for continuous morphologies.

The optimizer changes base roll/pitch/height and the 12 leg joints. Foot X/Y are
not prescribed; only the four foot soles are constrained to the ground plane.
Run from codesign/.
"""

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/motion_tracking"))

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from pd_replay import LEGS, build_model, rp_quat

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--designs", type=int, default=3)
parser.add_argument("--poses", type=int, default=32)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--source", nargs="+", default=glob.glob("runs/pipeline_v1/wide16_action/rollouts/e*/primitive_seed0.npz"))
parser.add_argument("--expert-config", type=Path, default=ROOT / "runs/pipeline_v1/experts_wide16_v2/walkForwardSlow__r00")
parser.add_argument("--settle-seconds", type=float, default=0.2)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
if args.designs < 3 or args.poses < 2 or args.settle_seconds < 0:
    parser.error("designs >= 3, poses >= 2 and settle-seconds >= 0 required")

configs = sorted(args.expert_config.glob("wandb/offline-run-*/files/config.pkl"))
if not configs:
    raise FileNotFoundError(f"No expert config under {args.expert_config}")
with configs[-1].open("rb") as f:
    cfg = pickle.load(f)
choice = cfg.get("choice")

rng = np.random.default_rng(args.seed)
seeds = []
for path in args.source:
    with np.load(path) as z:
        valid = np.argwhere(z["valid"])
        if len(valid):
            pick = valid[rng.choice(len(valid), min(128, len(valid)), replace=False)]
            seeds.extend(z["qpos"][b, t].copy() for b, t in pick)
if not seeds:
    raise ValueError("no valid source postures")

# Three coordinates are inert in the current arm-free A1 topology.
E = rng.uniform(0.05, 0.95, (args.designs, 13)).astype(np.float32)
E[:, [0, 11, 12]] = 0.5
# This experiment measures trained-shape sensitivity. Hold out poses within every E.
pose_split = np.full(args.poses, "test", dtype="<U5")
order = rng.permutation(args.poses)
n_train = max(1, round(0.70 * args.poses))
n_val = max(1, round(0.15 * args.poses))
pose_split[order[:n_train]] = "train"
pose_split[order[n_train:n_train + n_val]] = "val"


def audit(model, data, feet, floor):
    mujoco.mj_forward(model, data)
    foot_z = data.geom_xpos[feet, 2] - model.geom_size[feet, 0]
    contact = np.zeros(4, bool)
    nonfoot_pen, self_pen = 0.0, 0.0
    foot_index = {int(g): i for i, g in enumerate(feet)}
    for c in data.contact[:data.ncon]:
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            if other in foot_index and c.dist <= 1e-3:
                contact[foot_index[other]] = True
            elif c.dist < 0:
                nonfoot_pen = max(nonfoot_pen, -float(c.dist))
        elif c.dist < 0:
            self_pen = max(self_pen, -float(c.dist))
    joint_bad = np.any((data.qpos[7:19] < model.jnt_range[1:, 0] - 1e-6) |
                       (data.qpos[7:19] > model.jnt_range[1:, 1] + 1e-6))
    xy = data.geom_xpos[feet, :2]
    try:
        hull = ConvexHull(xy)
        com = data.subtree_com[0, :2]
        stable = bool(np.all(hull.equations[:, :2] @ com + hull.equations[:, 2] <= 1e-4))
    except Exception:  # Degenerate coincident feet are not a support polygon.
        stable = False
    valid = (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and not joint_bad and
             np.max(np.abs(foot_z)) <= 1.5e-3 and contact.all() and nonfoot_pen <= 2e-3 and
             self_pen <= 2e-3 and stable)
    return dict(valid=bool(valid), contact=contact, foot_z=foot_z, nonfoot_pen=nonfoot_pen,
                self_pen=self_pen, stable=stable, joint_bad=bool(joint_bad))


rows = []
for design_id, uniforms in enumerate(E):
    model = build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                        torque_limit=cfg.get("torque_limit"), uniforms=uniforms)
    feet = np.array([model.geom(f"{leg}_foot").id for leg in LEGS])
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    lo, hi = model.jnt_range[1:].T
    accepted, attempts = 0, 0
    while accepted < args.poses and attempts < args.poses * 40:
        attempts += 1
        source = seeds[rng.integers(len(seeds))]
        q_seed = np.clip(source[7:19] + rng.normal(0, 0.08, 12), lo + 1e-4, hi - 1e-4)
        roll_pitch = rng.normal(0, 0.08, 2)
        data = mujoco.MjData(model)

        def qpos_of(x):
            q = np.zeros(model.nq)
            q[:3], q[3:7], q[7:19] = [0, 0, x[2]], rp_quat(x[0], x[1]), x[3:]
            return q

        def residual(x):
            data.qpos[:] = qpos_of(x)
            mujoco.mj_kinematics(model, data)
            foot_z = data.geom_xpos[feet, 2] - model.geom_size[feet, 0]
            return np.r_[40 * foot_z, 0.6 * (x[3:] - q_seed), 0.3 * (x[:2] - roll_pitch)]

        x0 = np.r_[roll_pitch, source[2], q_seed]
        fit = least_squares(residual, x0, bounds=(np.r_[[-0.35, -0.35, 0.12], lo],
                                                  np.r_[[0.35, 0.35, 0.8], hi]),
                            max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        data.qpos[:] = qpos_of(fit.x)
        data.qvel[:] = 0
        data.ctrl[:12] = data.qpos[7:19]
        initial = audit(model, data, feet, floor)
        if not initial["valid"]:
            continue
        q0 = data.qpos.copy()
        settle_steps = round(args.settle_seconds / model.opt.timestep)
        fell = False
        for _ in range(settle_steps):
            mujoco.mj_step(model, data)
            w, x, y, _ = data.qpos[3:7]
            if not np.isfinite(data.qpos).all() or 1 - 2 * (x * x + y * y) < 0.5:
                fell = True
                break
        drift = float(np.linalg.norm(data.qpos[:3] - q0[:3]))
        if fell or drift > 0.08:
            continue
        rows.append((design_id, pose_split[accepted], q0, np.zeros(model.nv), initial["contact"],
                     initial["foot_z"], drift, attempts))
        accepted += 1
    print(f"design {design_id:03d}: {accepted}/{args.poses} accepted in {attempts} attempts", flush=True)
    if accepted < args.poses:
        raise RuntimeError(f"design {design_id}: only {accepted}/{args.poses} valid foot-anchored states")

args.out.parent.mkdir(parents=True, exist_ok=True)
design_id = np.array([r[0] for r in rows], np.int32)
np.savez_compressed(args.out, E=E[design_id], design_E=E, design_id=design_id,
                    split=np.array([r[1] for r in rows]), qpos=np.stack([r[2] for r in rows]),
                    qvel=np.stack([r[3] for r in rows]), contact=np.stack([r[4] for r in rows]),
                    foot_z=np.stack([r[5] for r in rows]), settle_drift_m=np.array([r[6] for r in rows]),
                    attempts=np.array([r[7] for r in rows]), topology=choice, seed=args.seed,
                    inert_E_dims=np.array([0, 11, 12], np.int32), settle_seconds=args.settle_seconds)
print(f"saved {len(rows)} states for {args.designs} morphologies -> {args.out}")
