"""Direct closed-loop transfer of frozen CALM HLC+LLC to TT e000 in MJX.

This is a feasibility pilot, NOT motion tracking, PPO training, or TT training.
The action is the A1 CALM joint target (FL,FR,RL,RR), mapped to TT joint DOFs.
Observations are TT rigid bodies mapped to CALM's 23-body convention.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
LEGS = ("FL", "FR", "RL", "RR")
TT_LEG = {"FL": "left_leg", "FR": "right_leg", "RL": "left_leg_1", "RR": "right_leg_1"}
CALM_BODIES = ["base", "trunk"] + [
    f"{leg}_{part}" for leg in LEGS
    for part in ("hip", "thigh_shoulder", "thigh", "calf", "foot")
] + ["imu_link"]


def body_mapping():
    """Name-based 23-slot observation adapter (no A1 geometry is transplanted)."""
    root = "quadruped_manipulator/quadruped_body"
    result = {"base": root, "trunk": root, "imu_link": root}
    for leg in LEGS:
        family = TT_LEG[leg]
        stem = f"quadruped_manipulator/{family}"
        part = "left_leg" if leg.endswith("L") else "right_leg"
        result[f"{leg}_hip"] = stem + "/" + part
        # CALM has a separate shoulder marker; TT does not. This is an explicit proxy.
        result[f"{leg}_thigh_shoulder"] = stem + "/" + part
        for item in ("thigh", "calf", "foot"):
            result[f"{leg}_{item}"] = stem + "/" + item
    return result


def check_body_mapping(model):
    import mujoco
    mapping = body_mapping()
    missing = [name for name in mapping.values()
               if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0]
    if missing:
        raise RuntimeError(f"TT/CALM observation mapping missing bodies: {missing}")
    assert len(CALM_BODIES) == 23 and set(CALM_BODIES) == set(mapping)
    return mapping


def physical_body_states(model, data, mapping):
    import mujoco
    values = []
    for label in CALM_BODIES:
        body = model.body(mapping[label]).id
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, vel, 0)
        values.append(np.r_[data.xpos[body], data.xquat[body][[1, 2, 3, 0]],
                            vel[3:], vel[:3]])
    result = np.asarray(values)
    if result.shape != (23, 13) or not np.isfinite(result).all():
        raise RuntimeError("Invalid TT body states for CALM")
    return result


def joint_addresses(model):
    from pd_replay import TT_LEG as pd_tt_leg
    names = [f"quadruped_manipulator/{pd_tt_leg[leg]}/{part}_joint"
             for leg in LEGS for part in ("hip", "thigh", "calf")]
    qa = np.array([model.jnt_qposadr[model.joint(name).id] for name in names])
    va = np.array([model.jnt_dofadr[model.joint(name).id] for name in names])
    if len(set(qa)) != 12 or len(set(va)) != 12:
        raise RuntimeError("Nonunique TT joint mapping")
    return names, qa, va


def make_model():
    """Keep the actual TT e000 inertia/geometry; replace its drives by CALM PD."""
    import mujoco
    from pd_replay import build_model
    model = build_model("00000011", uniforms=[0.5] * 13, torque_limit=33.5)
    model.opt.timestep = 1 / 600
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.impratio = 1.0
    # Disable native TT position actuators: otherwise torques would be counted twice.
    model.actuator_gainprm[:] = 0
    model.actuator_biasprm[:] = 0
    model.dof_damping[:] = 0
    model.dof_armature[:] = 0
    model.dof_frictionloss[:] = 0
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            model.geom_contype[gid], model.geom_conaffinity[gid] = 2, 1
        else:
            model.geom_contype[gid], model.geom_conaffinity[gid] = 1, 2
        model.geom_friction[gid] = [1.0, .005, .0001]
    check_body_mapping(model)
    joint_addresses(model)
    return model


def foot_contacts(model, d):
    """Contact flags read from MJX contact buffer, not reference labels."""
    import mujoco
    pairs = np.asarray(d.contact.geom)
    distances = np.asarray(d.contact.dist)
    floor = set(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist())
    foot = [model.geom(f"{leg}_foot").id for leg in ("FR", "FL", "RR", "RL")]
    flags = np.zeros(4, dtype=bool)
    for i, f in enumerate(foot):
        flags[i] = any(
            float(dist) <= .001 and
            ((int(a) in floor and int(b) == f) or
             (int(b) in floor and int(a) == f))
            for (a, b), dist in zip(pairs, distances)
        )
    return flags


def foot_positions(model, data):
    return np.stack([data.geom_xpos[model.geom(f"{l}_foot").id].copy()
                     for l in ("FR", "FL", "RR", "RL")])


def metrics(qpos, qvel, contact, foot_pos, commands, dt, reason, speed_threshold):
    n = len(commands)
    assert qpos.shape[0] == qvel.shape[0] == contact.shape[0] == foot_pos.shape[0] == n + 1
    vel = qvel[1:, :2]
    vel_error = float(np.sqrt(np.mean(np.sum((vel - commands[:, :2])**2, axis=1)))) if n else None
    slip_mask = contact[:-1] & contact[1:]
    slip = np.linalg.norm(np.diff(foot_pos[:, :, :2], axis=0), axis=-1) / dt
    slip_mean = float(slip[slip_mask].mean()) if slip_mask.any() else None
    up = 1 - 2 * (qpos[:, 4]**2 + qpos[:, 5]**2)
    survival = reason == "duration" and n == round(10 / dt)
    return dict(
        seconds=round(n * dt, 6), transitions=n, termination=reason, resets=0,
        survived_10s=bool(survival),
        velocity_rmse_mps=vel_error,
        speed_screening_threshold_mps=speed_threshold,
        screening_pass=bool(survival and vel_error is not None and vel_error <= speed_threshold),
        path_length_m=float(np.linalg.norm(np.diff(qpos[:, :2], axis=0), axis=1).sum()),
        displacement_xy_m=(qpos[-1, :2] - qpos[0, :2]).tolist(),
        min_up=float(up.min()),
        slip_contact_foot_center_mps=slip_mean,
        contact_fraction=float(contact.mean()),
        # Contact-point slip is NOT identical to foot-center displacement.
        slip_definition="foot-center XY speed while foot-floor contact holds at both endpoints",
    )


def run_episode(model, mjx_model, policy, joint_names, qa, va, mapping, commands,
                dt, torque_limit):
    import jax
    import jax.numpy as jnp
    import mujoco
    from mujoco import mjx

    va_jax = jnp.asarray(va)
    qa_jax = jnp.asarray(qa)
    @jax.jit
    def advance(state, target):
        def substep(_, s):
            tau = jnp.clip(40 * (target - s.qpos[qa_jax]) -
                           1.5 * s.qvel[va_jax], -torque_limit, torque_limit)
            force = jnp.zeros(model.nv).at[va_jax].set(tau)
            return mjx.step(mjx_model, s.replace(qfrc_applied=force))
        return jax.lax.fori_loop(0, 20, substep, state)

    cpu = mujoco.MjData(model)
    cpu.qpos[:] = model.key("home").qpos
    cpu.qvel[:] = 0
    cpu.qpos[:2] = 0
    mujoco.mj_forward(model, cpu)
    state = mjx.put_data(model, cpu)
    positions = [cpu.qpos.copy()]
    velocities = [cpu.qvel.copy()]
    feet = [foot_positions(model, cpu)]
    contacts = [foot_contacts(model, state)]
    targets, saved_commands, tau_starts = [], [], []
    reason = "duration"
    for command in commands:
        physical = physical_body_states(model, cpu, mapping)
        target = np.asarray(policy(physical, CALM_BODIES, command), dtype=np.float64)
        if target.shape != (12,) or not np.isfinite(target).all():
            reason = "invalid_policy_action"
            break
        tau0 = np.clip(40 * (target - np.asarray(state.qpos)[qa]) -
                       1.5 * np.asarray(state.qvel)[va], -torque_limit, torque_limit)
        state = advance(state, jnp.asarray(target))
        qp, qv = np.asarray(state.qpos), np.asarray(state.qvel)
        if not np.isfinite(qp).all() or not np.isfinite(qv).all():
            reason = "nonfinite_state"
            break
        cpu.qpos[:] = qp
        cpu.qvel[:] = qv
        mujoco.mj_forward(model, cpu)
        targets.append(target.copy())
        tau_starts.append(tau0.copy())
        saved_commands.append(np.asarray(command).copy())
        positions.append(qp.copy())
        velocities.append(qv.copy())
        feet.append(foot_positions(model, cpu))
        contacts.append(foot_contacts(model, state))
        up = 1 - 2 * (qp[4]**2 + qp[5]**2)
        if qp[2] < .06 or up < 0:
            reason = "trunk_fall"
            break

    qpos, qvel = np.asarray(positions), np.asarray(velocities)
    n = len(targets)
    result = dict(
        qpos=qpos, qvel=qvel, foot_pos=np.asarray(feet),
        contact=np.asarray(contacts, dtype=bool),
        target_q=np.asarray(targets).reshape(n, 12),
        action=np.asarray(targets).reshape(n, 12),
        initial_pd_torque=np.asarray(tau_starts).reshape(n, 12),
        command=np.asarray(saved_commands).reshape(n, 3),
        E=np.full(13, .5), choice=np.asarray("00000011"),
        joint_names=np.asarray(joint_names),
        time=np.arange(n + 1) * dt,
        valid=np.ones(n, dtype=bool),
        dt=np.asarray(dt),
    )
    if any(not np.isfinite(result[k]).all()
           for k in ("qpos", "qvel", "foot_pos", "target_q", "command")):
        raise RuntimeError("Nonfinite data persisted")
    assert result["qpos"].shape == (n + 1, model.nq)
    assert result["qvel"].shape == (n + 1, model.nv)
    assert result["contact"].shape == (n + 1, 4)
    assert result["target_q"].shape == (n, 12)
    return result, reason


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--wheel", type=Path,
                   default=ROOT / "runs/menagerie_calm_check_final/runtime/rl_games-1.1.4-py3-none-any.whl")
    p.add_argument("--episodes", type=int, default=1, help="1 smoke test, 10 for balanced command coverage")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--torque-limit", type=float, default=15.0,
                   help="CALM checkpoint's original joint torque limit, N*m")
    p.add_argument("--speed-threshold", type=float, default=.15,
                   help="Diagnostic world XY velocity RMSE threshold (m/s), NOT TT task success")
    args = p.parse_args()
    if not 1 <= args.episodes <= 10 or args.seed < 0:
        p.error("Use 1..10 episodes and a nonnegative seed")
    if not 0 < args.torque_limit <= 33.5 or args.speed_threshold <= 0:
        p.error("Invalid torque or speed threshold")
    if not args.wheel.is_file():
        p.error(f"rl_games wheel missing: {args.wheel}")
    if args.out.exists():
        p.error(f"Output already exists; results will not be overwritten: {args.out}")
    for f in ("lowLevelController.pth", "highLevelPolicy.pth"):
        ck = WORKSPACE / "external/a1-calm-main/checkpoints" / f
        if not ck.is_file():
            p.error(f"CALM checkpoint missing: {ck}")

    import jax
    import mujoco
    from mujoco import mjx
    from check_menagerie_mjx_policy import load_policy
    from eval_menagerie_calm import balanced_commands

    # Strict load of original, frozen A1 CALM HLC+LLC. No PPO and no retargeting.
    policy, source_joints = load_policy(WORKSPACE, args.wheel.resolve())
    expected = [f"{leg}_{part}_joint" for leg in LEGS for part in ("hip", "thigh", "calf")]
    if source_joints != expected:
        raise RuntimeError(f"Unexpected CALM joint order: {source_joints}")
    model = make_model()
    mapping = check_body_mapping(model)
    joint_names, qa, va = joint_addresses(model)
    mjx_model = mjx.put_model(model)
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for ep in range(args.episodes):
        commands, segments = balanced_commands(args.seed, ep)
        d, reason = run_episode(model, mjx_model, policy, joint_names,
                                qa, va, mapping, commands, 1 / 30, args.torque_limit)
        np.savez_compressed(args.out / f"episode_{ep:03d}.npz", **d)
        result = metrics(d["qpos"], d["qvel"], d["contact"], d["foot_pos"],
                         d["command"], 1 / 30, reason, args.speed_threshold)
        result.update(episode=ep, segments=segments,
                      max_initial_pd_torque_nm=float(np.abs(d["initial_pd_torque"]).max())
                      if len(d["initial_pd_torque"]) else None)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    hashes = {f: hashlib.sha256(
        (WORKSPACE / "external/a1-calm-main/checkpoints" / f).read_bytes()
    ).hexdigest() for f in ("lowLevelController.pth", "highLevelPolicy.pth")}
    summary = dict(
        status="SCREENING_PASS" if all(r["screening_pass"] for r in results) else "NOT_APPROVED",
        source="Frozen CALM A1 HLC+LLC -> TT 00000011 e000 -> MJX feedback",
        note="Diagnostic only. 10s survival is not command-following approval; no PPO/retargeting.",
        observations="23 slots; TT hip/thigh shoulder proxies differ from CALM A1 geometry",
        actions="Original CALM joint targets, name mapped to TT; PD 40/1.5 at 600Hz, torque clipped",
        dynamics="actual MJX qpos/qvel/contact (T+1), joint target (T), no resets",
        contact="MJX contact buffer: foot vs plane",
        choice="00000011", design="e000", E=[.5]*13,
        control_dt=1/30, physics_dt=1/600, substeps=20,
        kp=40, kd=1.5, torque_limit_nm=args.torque_limit,
        original_tt_actuators_disabled=True, self_collision=False,
        floor_collision=True, episodes=args.episodes, seed=args.seed,
        checkpoint_sha256=hashes, jax_devices=str(jax.devices()),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        results=results,
    )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    rows = "".join(
        "<tr><td>" + str(r["episode"]) + "</td><td>" +
        html.escape(r["termination"]) + "</td><td>" +
        f'{r["seconds"]:.2f}' + " s</td><td>" +
        ("-" if r["velocity_rmse_mps"] is None else f'{r["velocity_rmse_mps"]:.3f}') +
        "</td><td>" + ("PASS" if r["screening_pass"] else "FAIL") + "</td></tr>"
        for r in results
    )
    page = ("<!doctype html><html lang='ko'><meta charset='utf-8'>"
            "<title>CALM direct to TT e000</title><body style='max-width:900px;margin:auto;font:17px system-ui'>"
            "<h1>CALM HLC+LLC 직접 TT e000 실행</h1>"
            "<p>원본 체크포인트 동결, 30Hz CALM 폐루프, 600Hz MJX, 별도 PPO/리타겟팅 없음.</p>"
            "<p>몸체 23개 관측 중 CALM에만 있는 marker는 TT body로 대체했으므로 A1과 완전 동일한 관측은 아님.</p>"
            "<p>10초 생존과 실제 명령 속도 추종을 구별. 속도 RMSE 통과 기준 0.15 m/s는 예비 선별용.</p>"
            "<table border='1' cellpadding='8'><tr><th>Episode</th><th>종료</th><th>시간</th>"
            "<th>XY 속도 RMSE (m/s)</th><th>선별</th></tr>" + rows + "</table>"
            "<p>실제 물리 궤적/행동: episode_*.npz · 상세: summary.json</p></body></html>")
    (args.out / "report.html").write_text(page, encoding="utf-8")
    print("Saved:", args.out / "summary.json", "status:", summary["status"])


if __name__ == "__main__":
    main()
