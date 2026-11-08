"""Reward-gradient TT co-design and physical CPU MuJoCo evaluation for three 10s tasks.

Uses the existing joint diffusion sampler and frozen TT checkpoint. No MJX physics.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/motion_tracking"))
sys.path.insert(0, str(ROOT.parent / "transformer-transformer"))

import mujoco
import numpy as np
import torch

from pd_replay import build_model
from joint_diffusion_guidance import joint_diffusion_full_reward
from rollout_data import features
from tt_model import make_net, schedule, check_rest_checkpoint
from task_reward import StraightReward, ZigzagReward, TrackReward
from s0_geometry import s0_to_q

FEET = ("FR", "FL", "RR", "RL")
DT = 0.02
STEPS = 500


def floor_contacts(model, data, foot_ids, floor_id):
    out = np.zeros(4, np.float32)
    for contact in data.contact[:data.ncon]:
        if contact.dist > 1e-3:
            continue
        for j, f in enumerate(foot_ids):
            if (contact.geom1 == floor_id and contact.geom2 == f or
                    contact.geom2 == floor_id and contact.geom1 == f):
                out[j] = 1.0
    return out


def make_physics(e):
    model = build_model("00000011", uniforms=np.asarray(e, dtype=np.float64),
                        kp=30, torque_limit=33.5, flat_ground=True)
    model.opt.timestep = .005
    if model.nq != 19 or model.nv != 18 or model.nu < 12:
        raise ValueError("Unexpected model topology")
    return model


def settled_start(e):
    model = make_physics(e)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[:] = model.keyframe("home").qpos
    data.qpos[2] = 1.0
    feet = [model.geom(leg+"_foot").id for leg in FEET]
    floor = model.geom("floor").id
    mujoco.mj_forward(model, data)
    clear = data.geom_xpos[feet, 2] - model.geom_size[feet, 0]
    data.qpos[2] -= float(clear.min())
    data.ctrl[:] = model.keyframe("home").ctrl
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=200)  # 1 second settle
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("Nonfinite settled initial state")
    return features(np.stack([data.qpos]), np.stack([data.qvel]),
                    np.stack([floor_contacts(model, data, feet, floor)]))[0]


def replay(e, trajectory):
    model = make_physics(e)
    data = mujoco.MjData(model)
    feet = [model.geom(leg+"_foot").id for leg in FEET]
    floor = model.geom("floor").id
    token = np.asarray(trajectory, np.float32)
    out = np.zeros((STEPS, 54), np.float32)
    valid = np.zeros(STEPS, bool)
    actions = token[:, 42:54]
    if token.shape != (STEPS, 54) or not np.isfinite(token).all():
        return out, valid, dict(initial_valid=False, invalid_reason="nonfinite_or_wrong_shape")
    if not np.isfinite(e).all() or np.min(e) < 0 or np.max(e) > 1:
        return out, valid, dict(initial_valid=False, invalid_reason="morphology_out_of_range")
    try:
        data.qpos[:], data.qvel[:] = s0_to_q(token[0, :42])
        mujoco.mj_forward(model, data)
    except (ValueError, FloatingPointError) as exc:
        return out, valid, dict(initial_valid=False, invalid_reason=str(exc))
    # Evaluate exactly the generated morphology and S0, without silently lifting.
    raw_clearance = data.geom_xpos[feet, 2] - model.geom_size[feet, 0]
    gravity = token[0, 5:8]
    upright = 1 - 2 * (data.qpos[4]**2 + data.qpos[5]**2)
    support = floor_contacts(model, data, feet, floor)
    init_good = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                     and .9 <= np.linalg.norm(gravity) <= 1.1
                     and np.min(raw_clearance) >= -.002 and upright > .5
                     and support.sum() >= 2 and data.qpos[2] > .07)
    if not init_good:
        return out, valid, dict(initial_valid=False, invalid_reason="invalid_initial_support",
                                initial_contacts=int(support.sum()),
                                min_clearance_m=float(np.min(raw_clearance)))
    # Keep the physical rollout open-loop: no per-step policy corrections.
    for t in range(STEPS):
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            break
        contact = floor_contacts(model, data, feet, floor)
        # Compute heading-frame features from all accumulated physical states below.
        valid[t] = True
        if 1-2*(data.qpos[4]**2+data.qpos[5]**2) < .5 or data.qpos[2] < .07:
            break
        if t == STEPS-1:
            break
        data.ctrl[:12] = np.clip(actions[t],
                                  model.actuator_ctrlrange[:12, 0],
                                  model.actuator_ctrlrange[:12, 1])
        if not np.isfinite(data.ctrl).all():
            valid[t] = False
            break
        if t == 0:
            states_q = [data.qpos.copy()]
            states_v = [data.qvel.copy()]
            states_c = [contact]
        mujoco.mj_step(model, data, nstep=4)
        states_q.append(data.qpos.copy())
        states_v.append(data.qvel.copy())
        states_c.append(floor_contacts(model, data, feet, floor))
    n = int(valid.sum())
    if n and "states_q" not in locals():
        states_q = [data.qpos.copy()]
        states_v = [data.qvel.copy()]
        states_c = [floor_contacts(model, data, feet, floor)]
    if n:
        while len(states_q) < n:
            states_q.append(states_q[-1])
            states_v.append(states_v[-1])
            states_c.append(states_c[-1])
        out[:n, :42] = features(np.asarray(states_q[:n]),
                                 np.asarray(states_v[:n]), np.asarray(states_c[:n]))
        out[:n, 42:] = actions[:n]
    return out, valid, dict(initial_valid=True, valid_seconds=n*DT, fell=n<STEPS,
                            final_distance_m=float(np.linalg.norm(out[max(n-1,0), :2])),
                            forward_m=float(out[max(n-1,0), 0]),
                            mean_forward_speed_m_s=float(out[max(n-1,0), 0]/max(n*DT, DT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=("zigzag","circle","straight"), required=True)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--scales", nargs="+", type=float, default=[0, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.candidates < 1 or any(s < 0 for s in args.scales) or args.out.exists():
        parser.error("Use positive candidates, nonnegative scales, and a NEW output directory")
    checkpoint = args.ckpt / "checkpoint.pt" if args.ckpt.is_dir() else args.ckpt
    ck = torch.load(checkpoint, map_location=args.device, weights_only=False)
    ta = ck["args"]
    check_rest_checkpoint(ta)
    if 2 not in ta["modes"]:
        raise ValueError("Mode 2 required for full-horizon reward guidance")
    if ta.get("pose", "abs") != "abs":
        raise ValueError("CPU generator requires absolute-pose TT")
    net = make_net(ta).to(args.device)
    net.load_state_dict(ck["net"])
    net.eval().requires_grad_(False)
    mu = torch.as_tensor(ck["mu"], device=args.device)
    sd = torch.as_tensor(ck["sd"], device=args.device)
    emu, esd = ((.5, .5) if ta["norm"] == "minmax" else (.5, .25))
    beta, abar = schedule(ta["diff_steps"], args.device)
    reward = (ZigzagReward() if args.task == "zigzag"
              else TrackReward() if args.task == "circle" else StraightReward())
    args.out.mkdir(parents=True, exist_ok=False)

    def s0_fn(e_batch):
        values = np.stack([settled_start(e) for e in e_batch.detach().cpu().numpy()])
        return torch.as_tensor(values, device=args.device, dtype=mu.dtype)

    outcomes = {}
    for scale in args.scales:
        pred, morph, audit = joint_diffusion_full_reward(
            net, reward, n=args.candidates, episode_states=STEPS,
            horizon=ta["horizon"], state_dim=42, action_dim=12, e_dim=13,
            mu=mu, sd=sd, beta=beta, abar=abar,
            emu=emu, esd=esd, guidance_scale=scale, clip=ta["clip"],
            seed=args.seed, s0_fn=s0_fn)
        pp = pred.detach().cpu().numpy()
        ee = morph.detach().cpu().numpy()
        rows = []
        physical = []
        masks = []
        for i, (E, trajectory) in enumerate(zip(ee, pp)):
            actual, mask, metadata = replay(E, trajectory)
            physical.append(actual)
            masks.append(mask)
            # Do not assign a fake zero reward to invalid S0.
            with torch.no_grad():
                if mask.any():
                    score, terms = reward(torch.tensor(actual[None]), torch.tensor(mask[None]))
                    score = float(score[0])
                    terms = {k:float(v[0]) for k,v in terms.items()}
                    if metadata.get('fell', False):
                        score -= 20.0
                        terms['early_termination_penalty'] = 20.0
                else:
                    score, terms = None, {}
            rows.append(dict(candidate=i, morphology=E.tolist(),
                             reward=score, reward_terms=terms, **metadata,
                             task_progress=(terms.get("r_progress") * reward.unit_scale if score is not None else None),
                             progress_unit=reward.unit))
        np.savez_compressed(args.out / f"scale_{scale:g}.npz",
                            pred=pp, E=ee, mujoco=np.stack(physical),
                            valid=np.stack(masks))
        outcomes[f"scale_{scale:g}"] = dict(rows=rows, gradient_audit=audit)
        print(f"{args.task} scale={scale:g}: {[(x['candidate'],x['reward'],x['valid_seconds'] if x['initial_valid'] else 'invalid') for x in rows]}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(
        dict(task=args.task, physics_backend="MuJoCo CPU", seconds=10,
             control_hz=50, seed=args.seed, tt_checkpoint=str(checkpoint),
             results=outcomes), indent=2, ensure_ascii=False))
    print("Saved:", args.out / "summary.json")


if __name__ == "__main__":
    main()
