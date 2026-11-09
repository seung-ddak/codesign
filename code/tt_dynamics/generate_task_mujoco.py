"""Reward-gradient TT co-design and physical CPU MuJoCo evaluation for three 10s tasks.

Uses the existing joint diffusion sampler and frozen TT checkpoint. No MJX physics.
"""
import argparse
import json
import math
import sys
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/motion_tracking"))
sys.path.insert(0, str(ROOT.parent / "transformer-transformer"))

import mujoco
import numpy as np
import torch

from joint_diffusion_guidance import joint_diffusion_full_reward
from rollout_data import features
from tt_model import make_net, schedule, check_rest_checkpoint
from task_reward import SlalomReward, StraightReward, TrackReward
from s0_geometry import s0_to_q

FEET = ("FR", "FL", "RR", "RL")
DT = 0.02
STEPS = 500
SUBSTEPS = 4
EVALUATION_VERSION = 2
TASKS = {"circle": TrackReward, "zigzag": SlalomReward, "straight": StraightReward}


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


def add_slalom_poles(spec, task):
    for index, (x, y) in enumerate(task.poles):
        half_height = task.pole_height / 2
        spec.worldbody.add_geom(
            name=f"slalom_pole_{index}", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            pos=[x, y, half_height], size=[task.pole_radius, half_height, 0],
            rgba=[0.85, 0.18, 0.12, 1], contype=1, conaffinity=1)


def make_physics(e, task=None):
    from pd_replay import build_model

    scene_fn = partial(add_slalom_poles, task=task) if isinstance(task, SlalomReward) else None
    model = build_model("00000011", uniforms=np.asarray(e, dtype=np.float64),
                        kp=30, torque_limit=33.5, flat_ground=True, scene_fn=scene_fn)
    model.opt.timestep = DT / SUBSTEPS
    if model.nq != 19 or model.nv != 18 or model.nu != 12:
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
    mujoco.mj_forward(model, data)
    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError("Nonfinite settled initial state")
    return features(np.stack([data.qpos]), np.stack([data.qvel]),
                    np.stack([floor_contacts(model, data, feet, floor)]))[0]


def replay(e, trajectory, *, task=None, state_trace=None):
    """Apply 500 commands and record 501 states, including the terminal state.

    Row t contains state before control t. The terminal row repeats the last
    control for the state/action reward tensor. The archive's action mask
    identifies the 500 real control slots independently of that padding.
    """
    token = np.asarray(trajectory, np.float32)
    out = np.zeros((STEPS + 1, 54), np.float32)
    valid = np.zeros(STEPS + 1, bool)

    def invalid(reason, **details):
        return out, valid, dict(initial_valid=False, invalid_reason=reason,
                                termination_reason="invalid_initial_state",
                                control_steps=0, valid_seconds=0.0, fell=False,
                                complete_10s=False, pole_contacts=[], **details)

    if token.shape != (STEPS, 54) or not np.isfinite(token).all():
        return invalid("nonfinite_or_wrong_shape")
    e = np.asarray(e)
    if e.shape != (13,) or not np.isfinite(e).all() or np.any((e < 0) | (e > 1)):
        return invalid("invalid_morphology")
    model = make_physics(e, task)
    data = mujoco.MjData(model)
    feet = [model.geom(leg+"_foot").id for leg in FEET]
    floor = model.geom("floor").id
    try:
        data.qpos[:], data.qvel[:] = s0_to_q(token[0, :42])
        mujoco.mj_forward(model, data)
    except (ValueError, FloatingPointError) as exc:
        return invalid(str(exc))

    def fallen():
        return 1 - 2 * (data.qpos[4]**2 + data.qpos[5]**2) <= .5 or data.qpos[2] <= .07

    raw_clearance = data.geom_xpos[feet, 2] - model.geom_size[feet, 0]
    support = floor_contacts(model, data, feet, floor)
    bad_warnings = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                    mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)

    def physics_failed():
        return (any(data.warning[warning].number for warning in bad_warnings) or
                not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all())

    if physics_failed():
        return invalid("initial_physics_error")
    if (not .9 <= np.linalg.norm(token[0, 5:8]) <= 1.1 or
            np.min(raw_clearance) < -.002 or fallen() or support.sum() < 2):
        return invalid("invalid_initial_support", initial_contacts=int(support.sum()),
                       min_clearance_m=float(np.min(raw_clearance)))

    pole_ids = {}
    if isinstance(task, SlalomReward):
        pole_ids = {model.geom(f"slalom_pole_{i}").id: i for i in range(len(task.poles))}
    pole_contacts = set()

    def record_pole_contacts():
        for contact in data.contact[:data.ncon]:
            if contact.dist > 0:
                continue
            for pole, other in ((contact.geom1, contact.geom2), (contact.geom2, contact.geom1)):
                if pole in pole_ids and other >= 0 and model.geom_bodyid[other] > 0:
                    pole_contacts.add(pole_ids[pole])

    limited = model.actuator_ctrllimited[:12].astype(bool)
    actions = np.clip(token[:, 42:54],
                      np.where(limited, model.actuator_ctrlrange[:12, 0], -np.inf),
                      np.where(limited, model.actuator_ctrlrange[:12, 1], np.inf))
    qpos, qvel, contacts = [data.qpos.copy()], [data.qvel.copy()], [support]
    applied = []
    reason = "completed_horizon"
    if pole_ids:
        record_pole_contacts()
    for t, action in enumerate(actions):
        data.ctrl[:12] = action
        if pole_ids:
            for _ in range(SUBSTEPS):
                mujoco.mj_step(model, data)
                record_pole_contacts()
        else:
            mujoco.mj_step(model, data, nstep=SUBSTEPS)
        # A MuJoCo safety reset can replace invalid state with finite values.
        if physics_failed() or not math.isclose(data.time, (t + 1) * DT, rel_tol=0, abs_tol=1e-8):
            reason = "physics_error"
            break
        # Refresh geometry/contact at the post-step pose, including the final one.
        mujoco.mj_forward(model, data)
        if physics_failed():
            reason = "physics_error"
            break
        if pole_ids:
            record_pole_contacts()
        applied.append(data.ctrl[:12].copy())
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
        contacts.append(floor_contacts(model, data, feet, floor))
        if fallen():
            reason = "fall"
            break

    n = len(applied)
    valid[:n+1] = True
    out[:n+1, :42] = features(np.asarray(qpos), np.asarray(qvel), np.asarray(contacts))
    if n:
        out[:n, 42:] = applied
        out[n, 42:] = applied[-1]
    if state_trace is not None:
        state_trace.extend(qpos)
    seconds = n * DT
    forward = float(out[n, 0])
    return out, valid, dict(
        initial_valid=True, initial_contacts=int(support.sum()),
        min_clearance_m=float(np.min(raw_clearance)), termination_reason=reason,
        control_steps=n, valid_seconds=seconds, fell=reason == "fall",
        complete_10s=reason == "completed_horizon", pole_contacts=sorted(pole_contacts),
        final_distance_m=float(np.linalg.norm(out[n, :2])), forward_m=forward,
        mean_forward_speed_m_s=forward / max(seconds, DT))


def score_replay(task, actual, valid, meta):
    """Physical score plus task-specific success, independent of survival alone."""
    if not meta["control_steps"]:
        return dict(reward=None, reward_terms={}, progress=None, progress_unit=task.unit,
                    task_success=False, lane_compliance=None, gates_passed=None, finish_seconds=None,
                    gates_total=len(task.gates) if isinstance(task, SlalomReward) else None)
    with torch.no_grad():
        x = torch.as_tensor(actual[None])
        mask = torch.as_tensor(valid[None])
        value, components = task(x, mask)
        # Keep the action penalty's denominator in applied control steps.
        if "action_rate" in task.penalties:
            rate = task.p_action_rate(x[:, :-1], mask[:, 1:])
            value += task.penalties["action_rate"] * (components["p_action_rate"] - rate)
            components["p_action_rate"] = rate
    score = float(value[0])
    terms = {k: float(v[0]) for k, v in components.items()}
    if not meta["complete_10s"]:
        terms["early_termination_penalty"] = 20.0
        score -= terms["early_termination_penalty"]

    xy = actual[valid, :2]
    progress = terms["r_progress"] * task.unit_scale
    gates_passed, gates_total, finish_seconds = None, None, None
    if isinstance(task, SlalomReward):
        crossings = task.gate_crossings(xy)
        gates_passed, gates_total = len(crossings), len(task.gates)
        progress = meta["forward_m"]
        reached_goal = gates_passed == gates_total and not meta["pole_contacts"]
        next_distance = 0.0
        if gates_passed == gates_total:
            finish_seconds = crossings[-1] * DT
        else:
            next_distance = float(np.linalg.norm(xy[-1] - task.gates[gates_passed]))
        terms["gates_bonus"] = float(gates_passed)
        terms["next_gate_distance_penalty"] = min(next_distance, 2.0)
        terms["physical_collision_penalty"] = task.collision_penalty * len(meta["pole_contacts"])
        terms["finish_time_penalty"] = 1.0 if finish_seconds is None else finish_seconds / (STEPS * DT)
        score += terms["gates_bonus"] - (terms["next_gate_distance_penalty"] +
                                         terms["physical_collision_penalty"] +
                                         terms["finish_time_penalty"])
        lateral = np.abs(xy[:, 1] - task.y_ref(torch.as_tensor(xy[:, 0])).numpy())
    elif isinstance(task, StraightReward):
        lateral = np.abs(xy[:, 1])
        reached_goal = progress > 1e-6 and bool(np.all(lateral <= task.half_width + 1e-6))
    else:
        lateral = np.abs(np.linalg.norm(xy - task.center, axis=1) - task.radius)
        reached_goal = progress >= 1.0 - 1e-6 and bool(np.all(lateral <= task.half_width + 1e-6))
    return dict(reward=score if math.isfinite(score) else None, reward_terms=terms,
                progress=progress, progress_unit=task.unit,
                task_success=bool(meta["complete_10s"] and reached_goal and math.isfinite(score)),
                lane_compliance=float(np.mean(lateral <= task.half_width + 1e-6)),
                gates_passed=gates_passed, gates_total=gates_total, finish_seconds=finish_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", choices=("zigzag","circle","straight"), required=True)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--scales", nargs="+", type=float, default=[0, 1])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema",
                        help="use EMA weights for generation when present (raw for ablations)")
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
    use_ema = args.weights == "ema" and ck.get("ema_net") is not None
    if args.weights == "ema" and not use_ema:
        print("Checkpoint has no EMA weights; using raw model weights", flush=True)
    net.load_state_dict(ck["ema_net"] if use_ema else ck["net"])
    net.eval().requires_grad_(False)
    mu = torch.as_tensor(ck["mu"], device=args.device)
    sd = torch.as_tensor(ck["sd"], device=args.device)
    emu, esd = ((.5, .5) if ta["norm"] == "minmax" else (.5, .25))
    beta, abar = schedule(ta["diff_steps"], args.device)
    reward = TASKS[args.task]()
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
            actual, mask, metadata = replay(E, trajectory, task=reward)
            physical.append(actual)
            masks.append(mask)
            row = score_replay(reward, actual, mask, metadata)
            rows.append(dict(candidate=i, morphology=E.tolist(), **row, **metadata,
                             task_progress=row["progress"]))
        valid = np.stack(masks)
        np.savez_compressed(args.out / f"scale_{scale:g}.npz",
                            pred=pp, E=ee, mujoco=np.stack(physical), valid=valid,
                            action_valid=valid[:, 1:])
        outcomes[f"scale_{scale:g}"] = dict(rows=rows, gradient_audit=audit)
        print(f"{args.task} scale={scale:g}: {[(x['candidate'],x['reward'],x['valid_seconds'] if x['initial_valid'] else 'invalid') for x in rows]}", flush=True)
    (args.out / "summary.json").write_text(json.dumps(
        dict(task=args.task, evaluation_version=EVALUATION_VERSION,
             physics_backend="MuJoCo CPU", seconds=10,
             control_hz=50, seed=args.seed, tt_checkpoint=str(checkpoint),
             inference_weights="ema" if use_ema else "raw",
             results=outcomes), indent=2, ensure_ascii=False))
    print("Saved:", args.out / "summary.json")


if __name__ == "__main__":
    main()
