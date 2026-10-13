"""Reference-only jump comparison; no PPO, integration, or artifact mutation.

Run from codesign with ../transformer-transformer/.venv/bin/python
code/motion_tracking/validate_jump_retarget.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.signal import savgol_filter

from pd_replay import BASE_SG_WIN, LEGS, RADIUS, TT_LEG, build_model

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/TT_retarget/jump_validation"
CHOICES = ("00000000", "00000001", "00000010", "00000011")


def pre_ik_height(a, model, scale):
    """Reproduce retarget_tt's flat-ground body-height target, not the IK solve."""
    ma = build_model()
    da, d = mujoco.MjData(ma), mujoco.MjData(model)
    d.qpos[:] = model.key("home").qpos
    mujoco.mj_kinematics(model, d)
    base = model.jnt_bodyid[0]
    rot = d.xmat[base].reshape(3, 3)
    hips = np.array([(d.xpos[model.body(f"quadruped_manipulator/{TT_LEG[l]}/thigh").id]
                      - d.xpos[base]) @ rot for l in LEGS])
    feet = [ma.geom(f"{l}_foot").id for l in LEGS]
    thighs = [ma.body(f"{l}_thigh").id for l in LEGS]
    height = np.full(len(a["fix_q"]), np.nan)
    for i in range(len(height)):
        da.qpos[:] = np.r_[a["fix_base_pos"][i], a["fix_base_quat"][i], a["fix_q"][i]]
        mujoco.mj_kinematics(ma, da)
        ra = da.xmat[ma.jnt_bodyid[0]].reshape(3, 3)
        rel = (hips + scale * ((da.geom_xpos[feet] - da.xpos[thighs]) @ ra)) @ ra.T
        contact = a["fix_contact"][i].astype(bool)
        if contact.any():
            height[i] = (RADIUS - rel[contact, 2]).mean()
    valid = np.isfinite(height)
    az = a["fix_base_pos"][:, 2]
    offset = height - scale * az
    interpolated = scale * az + np.interp(np.arange(len(az)), np.flatnonzero(valid), offset[valid])
    return savgol_filter(interpolated, BASE_SG_WIN, 2)


def main():
    a = np.load(ROOT / "artifacts/pd_replay/jumpInPlace.npz")
    dt = float(a["dt"])
    az = a["fix_base_pos"][:, 2]
    time = np.arange(len(az)) * dt
    airborne = ~a["fix_contact"].astype(bool).any(axis=1)
    assert airborne.any(), "No A1 takeoff label; choose and document a different event."
    event = int(np.flatnonzero(airborne)[0])
    x = time - time[event]
    ah = az - az[event]
    peak = int(np.argmax(az))
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    diag, dax = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
    report = {"normalization": "z(t)-z(t_A1_first_no_contact); shared time axis, no phase warping",
              "takeoff_label_caveat": "Contact labels are fragmented; TT labels are cleaned. This is a source event, not independently verified physical takeoff.",
              "dt_s": dt, "a1_event_s": float(time[event]), "a1_peak_s": float(time[peak]),
              "a1_peak_above_event_mm": float(ah[peak] * 1000), "tt": {}}
    traces = {"time_s": time, "time_from_a1_event_s": x, "A1_height_mm": ah * 1000}
    for choice, ax in zip(CHOICES, axes.flat):
        z = np.load(ROOT / f"artifacts/retarget_tt/{choice}/jumpInPlace.npz")
        assert np.isclose(float(z["dt"]), dt) and len(z["fix_q"]) == len(az)
        np.testing.assert_allclose(z["a1_qpos"], np.c_[a["fix_base_pos"], a["fix_base_quat"], a["fix_q"]], atol=1e-7)
        model = build_model(choice)
        bz = z["fix_base_pos"][:, 2]
        bh = bz - bz[event]
        np.testing.assert_allclose(bh[event], 0, atol=1e-12)
        assert np.isfinite(z["fix_q"]).all() and np.isfinite(bz).all()
        pre = pre_ik_height(a, model, float(z["scale"]))
        err = np.linalg.norm(z["fix_foot_pos"] - z["fix_foot_target"], axis=2) * 1000
        np.testing.assert_allclose(err.mean(), float(z["fix_track_err"]) * 1000, atol=1e-6)
        lo, hi = model.jnt_range[1:].T
        margin = np.degrees(np.minimum(z["fix_q"] - lo, hi - z["fix_q"]))
        # Recompute FK to ensure saved feet and body/joint poses describe the same configuration.
        data = mujoco.MjData(model)
        feet = [model.geom(f"{l}_foot").id for l in LEGS]
        fk = []
        for pos, quat, q in zip(z["fix_base_pos"], z["fix_base_quat"], z["fix_q"]):
            data.qpos[:] = np.r_[pos, quat, q]
            mujoco.mj_kinematics(model, data)
            fk.append(data.geom_xpos[feet].copy() - [0, 0, RADIUS])
        np.testing.assert_allclose(fk, z["fix_foot_pos"], atol=1e-6)
        bp, pp = int(np.argmax(bz)), int(np.argmax(pre))
        tt_flight = ~z["fix_contact"].astype(bool).any(axis=1)
        window = (time >= 0.4) & (time <= 1.1)
        result = dict(peak_s=float(time[bp]), peak_delta_vs_a1_ms=float((bp-peak)*dt*1000),
                      pre_ik_peak_s=float(time[pp]), peak_above_common_event_mm=float(bh[bp]*1000),
                      height_rmse_jump_window_mm=float(np.sqrt(np.mean((bh[window]-ah[window])**2))*1000),
                      scale=float(z["scale"]), foot_error_mean_mm=float(err.mean()), foot_error_max_mm=float(err.max()),
                      body_ik_offset_max_mm=float(np.max(np.abs(bz-pre))*1000),
                      body_ik_offset_rms_mm=float(np.sqrt(np.mean((bz-pre)**2))*1000),
                      joint_limit_min_margin_deg=float(margin.min()), joint_limit_violation_samples=int((margin < -1e-6).sum()),
                      foot_lowest_z_mm=float(z["fix_foot_pos"][:, :, 2].min()*1000),
                      foot_target_lowest_z_mm=float(z["fix_foot_target"][:, :, 2].min()*1000),
                      tt_first_no_contact_s=float(time[np.flatnonzero(tt_flight)[0]]) if tt_flight.any() else None,
                      fk_consistency_pass=True)
        report["tt"][choice] = result
        traces[f"TT_{choice}_height_mm"] = bh * 1000
        ax.plot(x, ah*1000, color="darkorange", lw=2.3, label="A1 reference")
        ax.plot(x, bh*1000, color="royalblue", lw=2.3, label="TT retarget (post IK)")
        ax.plot(x, (pre-pre[event])*1000, color="royalblue", ls="--", alpha=.65, label="TT body target (pre IK)")
        ax.scatter([x[peak], x[bp]], [ah[peak]*1000, bh[bp]*1000], c=["darkorange", "royalblue"], zorder=5)
        ax.axvline(0, color="gray", ls=":")
        ax.axhline(0, color="gray", lw=.6)
        ax.set_title(f"TT {choice} | peak {result['peak_delta_vs_a1_ms']:+.0f} ms vs A1")
        ax.set_ylabel("Height above event baseline [mm]")
        ax.set_xlabel("Time from A1 first no-contact frame [s]")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        dax[0].plot(x, (bz-pre)*1000, label=choice)
        dax[1].plot(x, err.max(axis=1), label=choice)
        dax[2].plot(x, margin.min(axis=1), label=choice)
        dax[3].step(x, z["fix_contact"].sum(axis=1), where="post", label=choice, alpha=.7)
    dax[3].step(x, a["fix_contact"].sum(axis=1), where="post", color="darkorange", ls="--", label="A1")
    labels = ["Post IK - pre IK body z [mm]", "Max foot target error [mm]", "Min joint limit margin [deg]", "Contact-labelled feet"]
    for ax, label in zip(dax, labels):
        ax.set_ylabel(label)
        ax.axvline(0, color="gray", ls=":")
        ax.axhline(0, color="gray", lw=.6)
        ax.grid(alpha=.2)
        ax.legend(ncol=5, fontsize=8)
    dax[-1].set_xlabel("Time from A1 first no-contact frame [s]")
    fig.suptitle("Jump reference validation — no PPO / no dynamics rollout\nCommon A1 event; each body's own event height subtracted; timing unchanged")
    diag.suptitle("IK consistency and contact-label diagnostics — no PPO")
    for figure, name in ((fig, "height_comparison"), (diag, "ik_diagnostics")):
        figure.tight_layout(rect=(0, 0, 1, .95))
        figure.savefig(OUT / f"{name}.png", dpi=170)
        figure.savefig(OUT / f"{name}.svg")
        plt.close(figure)
    np.savetxt(OUT / "height_traces.csv", np.column_stack(list(traces.values())), delimiter=",", header=",".join(traces), comments="")
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
