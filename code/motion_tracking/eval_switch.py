"""목표 바꾸기 (전환) 평가: 모션 A 를 따라가다 정해진 시점에 목표를 B 로 바꾸고, B 를 얼마나 빨리 따라잡는지 잰다.

usage (codesign/ 에서):
  MUJOCO_GL=egl ../transformer-transformer/.venv/bin/python code/motion_tracking/eval_switch.py runs/TT_tracking/tt_switch/00000000
출력: <run>/switch_eval/switch_eval.csv, videos/<A>_to_<B>.mp4
통과 = 넘어지지 않음 + 바꾼 뒤 RECOVER_S 안에 관절 RMS < 5°, 발 < 3 cm, 자세 < 5° 를 HOLD 프레임 연속 유지 (eval_motions 기준과 같은 값).
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

# (A, B, 학습 여부). held-out = 학습에서 뺀 '회전 보행 → 점프'
PAIRS = [
    ("walkForwardSlow", "jumpInPlace", "trained"),
    ("jumpInPlace", "walkForwardSlow", "trained"),
    ("walkForwardFast", "sitDown", "trained"),
    ("trotForwardFast", "walkBackwardSlow", "trained"),
    ("squat", "trotLeftSlow", "trained"),
    ("walkLeftSlow", "tiltRoll", "trained"),
    ("walkCWSlow", "jumpInPlace", "held-out"),
    ("trotCCWSlow", "jumpInPlace", "held-out"),
    ("walkCCWSlow", "walkToJump", "held-out"),
]
SWITCH_S = 2.0  # A 를 따라가는 시간 (A 가 짧으면 끝 10 프레임 전). 이후 첫 '두 발 이상 접지' 프레임에서 바꾼다
AFTER_S = 3.0  # 바꾼 뒤 관찰 시간 (B 가 먼저 끝나면 거기까지)
RECOVER_S = 1.0
HOLD = 10  # 프레임 (0.2 s)
TH = dict(joint=5.0, foot=3.0, orn=5.0)  # °, cm, °

run = Path(sys.argv[1])
ckpt = sorted(glob.glob(str(run / "files/checkpoints/*")), key=lambda p: int(os.path.basename(p)))[-1]
cfg, policy = load_run(ckpt)
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
ecfg["switch_entry"] = os.environ.get("SWITCH_ENTRY", "start")  # B 진입: start = B 처음부터 (사용자 결정)
AT_END = os.environ.get("SWITCH_AT", "end") == "end"  # end = A 를 끝까지 따라간 뒤 B (DeepMimic 동작 경계)
choice = cfg.get("choice")
env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale")), **ecfg)
reset_with_traj, step, switch_now, policy = jax.jit(env.reset_with_traj), jax.jit(env.step), jax.jit(env.switch_now), jax.jit(policy)
clips = ecfg["clips"]

m = env.mj_model
d = mujoco.MjData(m)
m.vis.global_.offwidth, m.vis.global_.offheight = 768, 432
renderer = mujoco.Renderer(m, 432, 768)
vopt = mujoco.MjvOption()
vopt.geomgroup[3] = 1  # TT 형상 geom
cam = mujoco.MjvCamera()
cam.type, cam.distance, cam.azimuth, cam.elevation = mujoco.mjtCamera.mjCAMERA_FREE, 1.6, 120.0, -15.0

feet = np.array(env._foot_ids)
out = run / f"switch_eval_{ecfg['switch_entry']}_{'end' if AT_END else 'time'}"
(out / "videos").mkdir(parents=True, exist_ok=True)
src_ok = np.asarray(env.src_ok)
rows = []
for a, b, kind in PAIRS:
    ca, cb = clips.index(a), clips.index(b)
    start, end = int(env.clip_start[ca]), int(env.clip_end[ca])
    f_sw = end - 1 if AT_END else min(start + round(SWITCH_S / env.dt), end - 10)
    step_dir = -1 if AT_END else 1  # 두 발 이상 접지 프레임으로 (end 는 앞쪽으로 찾는다)
    while not src_ok[f_sw]:
        f_sw += step_dir
    rng = jax.random.PRNGKey(0)
    s = reset_with_traj(rng, jnp.int32(ca), jnp.int32(start))
    frames, errs, fell, k_sw = [], [], False, None
    for t in range(end - start + round(AFTER_S / env.dt) + 20):
        if k_sw is None and int(s.info["frame"]) >= f_sw:
            s = switch_now(s, jnp.int32(cb))
            k_sw = t
        rng, key = jax.random.split(rng)
        s = step(s, policy(s.obs, key)[0])
        i = int(s.info["frame"])
        qpos = np.asarray(s.data.qpos)
        ref_q, _ = env._ref(s.info["frame"], s.info["ref_off"])
        ref_q = np.asarray(ref_q)
        rot = np.asarray(s.data.xmat[env._base_idx])
        foot_b = (np.asarray(s.data.geom_xpos[feet]) - np.asarray(s.data.xpos[env._base_idx])) @ rot
        if k_sw is not None:
            errs.append((np.degrees(np.sqrt(np.mean((qpos[7:7 + env.n_ctrl] - ref_q[7:7 + env.n_ctrl]) ** 2))),
                         np.linalg.norm(foot_b - np.asarray(env.ref_foot[i]), axis=1).mean() * 100,
                         np.degrees(float(s.info["orn_err"]))))
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        cam.lookat[:] = qpos[:3]
        renderer.update_scene(d, camera=cam, scene_option=vopt)
        frames.append(renderer.render())
        if float(s.info["done/fellover"]) > 0:
            fell = True
            break
        if k_sw is not None and (i >= int(s.info["clip_end"]) - 1 or t - k_sw >= round(AFTER_S / env.dt)):
            break
    e = np.array(errs)
    good = (e[:, 0] < TH["joint"]) & (e[:, 1] < TH["foot"]) & (e[:, 2] < TH["orn"])
    run_len = np.convolve(good, np.ones(HOLD), "valid") >= HOLD  # HOLD 프레임 연속 통과 시작점
    rec = int(np.argmax(run_len)) if run_len.any() else None
    late = e[env.switch_window:] if len(e) > env.switch_window else e  # 보상 완화 구간 이후 평균
    r = dict(A=a, B=b, kind=kind, survived=int(not fell), recover_s=(rec * env.dt if rec is not None else -1),
             joint_deg=float(late[:, 0].mean()), foot_cm=float(late[:, 1].mean()), orn_deg=float(late[:, 2].mean()),
             observed_s=len(e) * env.dt)
    r["pass"] = int(r["survived"] and rec is not None and rec * env.dt <= RECOVER_S)
    rows.append(r)
    imageio.mimsave(out / "videos" / f"{a}_to_{b}.mp4", frames, fps=round(1 / env.dt))
    print(f"{a:16s} → {b:16s} {kind:8s} surv={r['survived']} recover={r['recover_s']:5.2f}s joint={r['joint_deg']:5.2f}° "
          f"foot={r['foot_cm']:5.2f}cm orn={r['orn_deg']:5.2f}° pass={r['pass']}")

with open(out / "switch_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("->", out, f"pass {sum(r['pass'] for r in rows)}/{len(rows)}")
