"""Run full 10-second MJX evaluation of a morphology-conditioned tracking PPO.

Use a checkpoint from train_rl_procedural.py, not the old e000-only checkpoint.
Evaluate the exact retargeted reference of each E, with no autoreset or PPO updates.
"""
import argparse
import functools
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "transformer-transformer"))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from eval_calm_mjx_tracking import check_reference_morphology
from eval_calm_pd_tracking import foot_contacts, tracking_errors
from multi_design import normalize_design_catalog
from pd_replay import LEGS, build_model
from policy_io import load_run
from track_env import TrackMjxEnv


def reference_root(cfg):
    # Training appends /<topology> to the base reference_dir before saving config.
    return Path(str(cfg.env.reference_dir)).resolve().parents[1]


def evaluate_one(cfg, policy, reference_root_dir, design, E, clip, seed):
    choice = str(cfg.choice).zfill(8)
    refdir = reference_root_dir / design / choice
    refpath = refdir / (clip + ".npz")
    if not refpath.is_file():
        raise FileNotFoundError(refpath)
    with np.load(refpath) as f:
        check_reference_morphology(f, E)
        nframes = len(f["fix_q"])
        if nframes < 301:
            raise ValueError(f"Need >=301 frames for a 10s test: {refpath}")

    env_args = OmegaConf.to_container(cfg.eval_env, resolve=True)
    env_args.pop("_target_", None)
    env_args.update(reference_dir=str(refdir), clips=[clip], max_episode_length=300)
    model_fn = functools.partial(
        build_model, choice=choice, uniforms=E, kp=cfg.kp,
        tau_scale=cfg.get("tau_scale"), torque_limit=cfg.get("torque_limit"),
    )
    env = TrackMjxEnv(mj_model_fn=model_fn, **env_args)
    if not np.isclose(env.dt, 1 / 30) or env.n_substeps != 10:
        raise ValueError("Expected native 30Hz control and 300Hz MJX physics")
    model = env.mj_model
    feet = np.array([model.geom(name + "_foot").id for name in LEGS])
    planes = set(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist())
    ghost = mujoco.MjData(model)
    morph_obs = jnp.asarray(E, dtype=jnp.float32)

    state = env.reset_with_traj(jax.random.PRNGKey(seed), jnp.int32(0), env.clip_start[0])
    ref_qpos_host = np.asarray(env.ref_qpos)

    def advance(s, key):
        obs = jnp.concatenate((s.obs, morph_obs))
        action = jnp.clip(policy(obs, key)[0], *env.clip_actions)
        next_state = env.step(s, action)
        return next_state, mjx.forward(env.mjx_model, next_state.data)

    advance = jax.jit(advance)
    key = jax.random.PRNGKey(seed + 101)
    qpos, refpos, footpos, reffootpos, slips = [], [], [], [], []
    fell = False
    invalid = False

    for i in range(300):
        key, subkey = jax.random.split(key)
        state, snapshot = advance(state, subkey)
        invalid = bool(np.asarray(state.info["state_is_nan"]) > 0)
        if invalid:
            break
        native = mjx.get_data(model, snapshot)
        native = native[0] if isinstance(native, (list, tuple)) else native
        r = ref_qpos_host[int(np.asarray(state.info["frame"]))]
        ghost.qpos[:] = r
        mujoco.mj_forward(model, ghost)
        _, slip = foot_contacts(model, native, feet, planes)
        qpos.append(native.qpos.copy())
        refpos.append(r.copy())
        footpos.append(native.geom_xpos[feet].copy())
        reffootpos.append(ghost.geom_xpos[feet].copy())
        slips.extend(np.asarray(slip)[np.isfinite(slip)].tolist())
        fell = bool(np.asarray(state.info["done/fellover"]) > 0)
        if bool(np.asarray(state.done) > 0):
            break

    n = len(qpos)
    row = dict(design=design, motion=clip, evaluated_steps=n, evaluated_seconds=n / 30,
               survived_10s=bool(n == 300 and not fell and not invalid),
               fell=bool(fell), nonfinite_state=bool(invalid))
    if n == 0:
        row["error"] = "No valid MJX states"
        return row
    q, r = np.asarray(qpos), np.asarray(refpos)
    fp, rf = np.asarray(footpos), np.asarray(reffootpos)
    rot = Rotation.from_quat(q[:, [4, 5, 6, 3]]).as_matrix()
    rr = Rotation.from_quat(r[:, [4, 5, 6, 3]]).as_matrix()
    f_body = np.einsum("tfi,tij->tfj", fp - q[:, None, :3], rot)
    r_body = np.einsum("tfi,tij->tfj", rf - r[:, None, :3], rr)
    row.update(tracking_errors(q, r, fp, rf))
    row["foot_body_mean_error_m"] = float(np.linalg.norm(f_body - r_body, axis=-1).mean())
    row["slip_mean_mps"] = float(np.mean(slips)) if slips else None
    row["supported_foot_samples"] = len(slips)
    row["tracking_error_screen_pass"] = bool(
        row["survived_10s"] and row["joint_rmse_deg"] < 5
        and row["foot_body_mean_error_m"] < .03
        and row["orientation_mean_deg"] < 5
    )
    return row


def write_html(rows, out):
    headers = ["design", "motion", "survived_10s", "evaluated_seconds",
               "joint_rmse_deg", "body_position_rmse_m",
               "foot_body_mean_error_m", "orientation_mean_deg", "slip_mean_mps"]
    lines = ["<!doctype html><html lang='ko'><meta charset='utf-8'>",
             "<title>Shared PPO · 10s MJX evaluation</title>",
             "<style>body{font:15px system-ui;max-width:1250px;margin:32px auto;padding:0 16px}"
             "table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccc}"
             "th{text-align:left;background:#eef3f6}</style>",
             "<h1>공유 PPO · 10초 MJX 실제 추종 평가</h1>",
             "<p>각 형상별 리타겟 Reference를 첫 프레임부터 재생하며 전도 후 오차는 누적하지 않음."
             " 조기 종료 실행의 작은 RMSE를 높은 성능으로 해석하지 말 것.</p>",
             "<table><thead><tr>" + "".join("<th>" + html.escape(k) + "</th>" for k in headers)
             + "</tr></thead><tbody>"]
    for row in rows:
        cells = []
        for k in headers:
            value = row.get(k)
            cells.append("<td>" + html.escape(f"{value:.4f}" if isinstance(value, float) else str(value)) + "</td>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    lines.append("</tbody></table></html>")
    (out / "report.html").write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--designs", nargs="+", help="Default: train + held-out designs")
    ap.add_argument("--clips", nargs="+", help="Default: all train and held-out clips")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        ap.error("Output directory is not empty; do not overwrite measurements")
    cfg, policy = load_run(args.checkpoint)
    if not cfg.get("multi_design"):
        ap.error("Checkpoint was not trained with multi_design=true")
    root = reference_root(cfg)
    catalog_path = root / "designs.json"
    if not catalog_path.is_file():
        catalog_path = root.parent / "designs.json"
    if not catalog_path.is_file():
        raise FileNotFoundError(catalog_path)
    catalog = normalize_design_catalog(json.loads(catalog_path.read_text()), str(cfg.choice).zfill(8))
    designs = list(dict.fromkeys(args.designs or (list(cfg.train_designs) + list(cfg.eval_designs))))
    clips = list(dict.fromkeys(args.clips or (list(cfg.env.clips) + list(cfg.eval_env.clips))))
    missing = [name for name in designs if name not in catalog]
    if missing:
        raise ValueError(f"Unknown morphologies in catalog: {missing}")
    for design in designs:
        for clip in clips:
            path = root / design / str(cfg.choice).zfill(8) / (clip + ".npz")
            if not path.is_file():
                raise FileNotFoundError(f"Cannot evaluate missing morphology-matched reference: {path}")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for design in designs:
        E = np.asarray(catalog[design], dtype=float)
        for clip in clips:
            row = evaluate_one(cfg, policy, root, design, E, clip, args.seed)
            row["train_design"] = design in cfg.train_designs
            row["train_motion"] = clip in cfg.env.clips
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.out / "conditions.json").write_text(
        json.dumps({"checkpoint": str(args.checkpoint.resolve()), "seed": args.seed,
                    "designs": designs, "clips": clips, "ctrl_dt": 1 / 30,
                    "physics_dt": 1 / 300, "source": "native MJX 10s first-frame evaluation"},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.out / "metrics.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_html(rows, args.out)


if __name__ == "__main__":
    main()
