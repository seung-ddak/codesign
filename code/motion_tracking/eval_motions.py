"""모션별 평가: 각 모션을 0 프레임부터 끝까지 정책으로 추종하고 reference(fix_*) 대비 오차를 잰다.

usage (codesign/ 에서):
  MUJOCO_GL=egl ../transformer-transformer/.venv/bin/python code/motion_tracking/eval_motions.py runs/A1_tracking/try2
출력: <run>/motion_eval/motion_eval.csv, videos/<motion>.mp4
영상: 파랑 = 학습 형상, 반투명 주황 = 레퍼런스 모션 (A1 원래 관절각). 넓은 화면, 몸통을 따라가는 카메라.
"""

import csv
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "transformer-transformer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import imageio
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from omegaconf import OmegaConf

from pd_replay import build_model
from t2.eval.stair_design import load_run
from track_env import TrackMjxEnv

# 통과 기준 (2026-10-11 사용자 확정). 접지 일치는 착지·이륙 타이밍 ±CONTACT_TOL 프레임 허용 (사용자 확정)
PASS = dict(joint_rms_deg=5.0, foot_err_cm=3.0, orn_err_deg=5.0, end_pos_frac=0.10, end_pos_min_m=0.05, contact_agree=0.90)
CONTACT_DIST = 1e-3  # m. 발 geom 과 바닥 거리가 이 이하면 접지
CONTACT_TOL = 1  # 프레임 (20 ms). reference 접지 자체가 50Hz 운동학 추정이라 전환 시점에 ±1 프레임 불확실
FOOT_R = 0.02  # 발 sphere 반지름

run = Path(sys.argv[1])
ckpt = sorted(glob.glob(str(run / "files/checkpoints/*")), key=lambda p: int(os.path.basename(p)))[-1]
cfg, policy = load_run(ckpt)
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
choice = cfg.get("choice")  # 없거나 "A1" 이면 A1 몸
choice = None if choice == "A1" else choice
env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale")), **ecfg)
reset_with_traj, step, policy = jax.jit(env.reset_with_traj), jax.jit(env.step), jax.jit(policy)

m = env.mj_model
d = mujoco.MjData(m)
m.vis.global_.offwidth, m.vis.global_.offheight = 768, 432
renderer = mujoco.Renderer(m, 432, 768)
m_ref = build_model()  # 레퍼런스 모션 = A1
d_ref = mujoco.MjData(m_ref)
vopt, pert = mujoco.MjvOption(), mujoco.MjvPerturb()  # A1 고스트용
vopt_sim = mujoco.MjvOption()
if choice is not None:
    # TT 형상 geom 은 group 3 (기본 숨김). A1 고스트는 메쉬가 TT 렌더 컨텍스트에 없으므로 group 3 충돌 primitive 로 그림
    vopt_sim.geomgroup[3] = 1
    vopt.geomgroup[:] = [0, 0, 0, 1, 0, 0]
cam = mujoco.MjvCamera()
cam.type, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 1.4, 120.0, -15.0
SIM_RGBA, REF_RGBA = np.array([0.25, 0.5, 0.95, 1.0]), np.array([1.0, 0.55, 0.1, 0.45])


def render(qpos, ref_qpos, a1_qpos):
    d.qpos[:] = qpos
    mujoco.mj_forward(m, d)
    d_ref.qpos[:] = a1_qpos
    d_ref.qpos[:2] = ref_qpos[:2]  # A1 은 형상 reference 의 수평 위치에 겹쳐 그림 (리타겟이 보폭을 다리 길이 비로 줄임)
    mujoco.mj_forward(m_ref, d_ref)
    cam.lookat[:] = ref_qpos[:3]  # 레퍼런스 몸통을 따라감
    renderer.update_scene(d, camera=cam, scene_option=vopt_sim)
    scn = renderer.scene
    n0 = scn.ngeom
    mujoco.mjv_addGeoms(m_ref, d_ref, vopt, pert, mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
    for k in range(scn.ngeom):
        g = scn.geoms[k]
        if g.category == mujoco.mjtCatBit.mjCAT_DYNAMIC:
            g.matid, g.rgba[:] = -1, (REF_RGBA if k >= n0 else SIM_RGBA)
    return renderer.render()

feet = np.array(env._foot_ids)
out = run / "motion_eval"
(out / "videos").mkdir(parents=True, exist_ok=True)

rows = []
only = sys.argv[2:]  # 모션 이름을 주면 그 모션만
for ci, name in enumerate(ecfg["clips"]):
    if only and name not in only:
        continue
    start, end = int(env.clip_start[ci]), int(env.clip_end[ci])
    a1 = np.load(os.path.join(ecfg["reference_dir"], f"{name}.npz"))
    a1_qpos = a1["a1_qpos"] if "a1_qpos" in a1.files else np.asarray(env.ref_qpos[start:end])
    rng = jax.random.PRNGKey(0)
    s = reset_with_traj(rng, jnp.int32(ci), jnp.int32(start))
    jerr, ferr, oerr, cagree, frames = [], [], [], [], []
    log = {k: [] for k in ("sim_contact", "ref_contact", "sim_foot_z", "ref_foot_z", "sim_foot_dist")}  # 접지 판정 분석용
    fell = False
    for _ in range(end - start - 1):
        rng, key = jax.random.split(rng)
        s = step(s, policy(s.obs, key)[0])
        i = int(s.info["frame"])
        qpos = np.asarray(s.data.qpos)
        ref_q = np.asarray(env.ref_qpos[i])
        jerr.append(np.sqrt(np.mean((qpos[7:7 + env.n_ctrl] - ref_q[7:7 + env.n_ctrl]) ** 2)))
        rot = np.asarray(s.data.xmat[env._base_idx])
        foot_b = (np.asarray(s.data.geom_xpos[feet]) - np.asarray(s.data.xpos[env._base_idx])) @ rot
        ferr.append(np.linalg.norm(foot_b - np.asarray(env.ref_foot[i]), axis=1).mean())
        oerr.append(float(s.info["orn_err"]))  # track_env: reference 대비 base 회전각 (rad)
        geom, dist = np.asarray(s.data.contact.geom), np.asarray(s.data.contact.dist)
        sim_c = np.array([((geom == f).any(1) & (dist < CONTACT_DIST)).any() for f in feet])
        ref_c = np.asarray(env.ref_contact[i]).astype(bool)
        log["sim_contact"].append(sim_c)
        log["ref_contact"].append(ref_c)
        log["sim_foot_z"].append(np.asarray(s.data.geom_xpos[feet, 2]) - FOOT_R)
        log["ref_foot_z"].append(a1["fix_foot_pos"][i - start, :, 2] if "fix_foot_pos" in a1.files else np.full(4, np.nan))
        log["sim_foot_dist"].append([dist[(geom == f).any(1)].min(initial=1.0) for f in feet])
        frames.append(render(qpos, np.asarray(env.ref_qpos[i]), a1_qpos[i - start]))
        if float(s.info["done/fellover"]) > 0:
            fell = True
            break
        if i >= end - 1:
            break
    sc, rc = np.array(log["sim_contact"]), np.array(log["ref_contact"])
    ok = np.zeros_like(sc)
    for k in range(-CONTACT_TOL, CONTACT_TOL + 1):  # sim 접지가 reference 의 t-1..t+1 중 하나와 같으면 일치
        idx = np.clip(np.arange(len(rc)) + k, 0, len(rc) - 1)
        ok |= sc == rc[idx]
    cagree = ok
    ref_end, ref_start = np.asarray(env.ref_qpos[end - 1, :3]), np.asarray(env.ref_qpos[start, :3])
    travel = float(np.linalg.norm((ref_end - ref_start)[:2]))
    end_err = float(np.linalg.norm(qpos[:3] - np.asarray(env.ref_qpos[i, :3])))
    end_tol = max(PASS["end_pos_frac"] * travel, PASS["end_pos_min_m"])
    inv_bad = float(np.load(os.path.join(ecfg["reference_dir"], f"{name}.npz"))["inv_bad_frac"])
    r = dict(
        motion=name, frames=end - start, survived=int(not fell),
        joint_rms_deg=float(np.degrees(np.mean(jerr))), foot_err_cm=float(np.mean(ferr) * 100),
        orn_err_deg=float(np.degrees(np.mean(oerr))), end_pos_err_cm=end_err * 100, end_pos_tol_cm=end_tol * 100,
        travel_m=travel, contact_agree=float(np.mean(cagree)), ref_infeasible=inv_bad,
    )
    r["pass_joint"] = int(r["joint_rms_deg"] < PASS["joint_rms_deg"])
    r["pass_foot"] = int(r["foot_err_cm"] < PASS["foot_err_cm"])
    r["pass_orn"] = int(r["orn_err_deg"] < PASS["orn_err_deg"])
    r["pass_end_pos"] = int(end_err < end_tol)
    r["pass_contact"] = int(r["contact_agree"] > PASS["contact_agree"])
    r["pass_all"] = int(r["survived"] and all(r[k] for k in r if k.startswith("pass_") and k != "pass_all"))
    rows.append(r)
    imageio.mimsave(out / "videos" / f"{name}.mp4", frames, fps=round(1 / env.dt))
    np.savez(out / f"{name}_frames.npz", **{k: np.array(v) for k, v in log.items()})
    print(f"{name:18s} surv={r['survived']} joint={r['joint_rms_deg']:5.2f}° foot={r['foot_err_cm']:5.2f}cm "
          f"orn={r['orn_err_deg']:5.2f}° end={r['end_pos_err_cm']:5.1f}/{r['end_pos_tol_cm']:4.1f}cm "
          f"contact={r['contact_agree']:.2f} ref_infeasible={inv_bad:.2f} pass={r['pass_all']}")

with open(out / "motion_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("->", out, f"pass {sum(r['pass_all'] for r in rows)}/{len(rows)}")
