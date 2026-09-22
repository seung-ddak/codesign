"""TT-form and RL-form comparison rendered side by side.

This is a comparison/debug script only. The main coevolution training remains
untouched. To isolate morphology-generation effects from the forward-running RL
reward, the default comparison objective is the TT tracking reward rather than
any forward-locomotion reward.

  TT : choices 00010000 seed 5563   RL : choices 00011001 seed 1298   (둘 다 traj 9 의 최고 실측)

  .venv/bin/python code/compare_tt_rl_sim.py --traj 9 --reward-fn tracking_only
"""

import argparse
import json
import os
import pickle

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import hydra
import imageio.v2 as imageio
import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

ROOT = "/home/cgr-ugrad-2026/quad_ws/codesign"
TT = "/home/cgr-ugrad-2026/quad_ws/transformer-transformer"
OUT = os.path.join(ROOT, "results")
os.chdir(TT)

from t2.env.track_env import TrackEnv

W, H = 1200, 620
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
# TT = 청록, RL = 주황. 로봇 지오메트리에 직접 입힌다.
COLORS = {"TT": (0.20, 0.72, 0.85), "RL": (0.98, 0.55, 0.15)}
Y_OFFSET = {"TT": 0.45, "RL": -0.45}


def build_runner(choices, expert_ckpt, traj_idx, pkl, reward_fn="tracking_only"):
    with initialize_config_dir(config_dir=f"{TT}/config", version_base="1.3"):
        cfg = compose(
            config_name="evaluate_hardware_opt_rl",
            overrides=[
                "ckpt_path=checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt",
                f"pickle_path={pkl}",
                f"reward_fns@eval_fn.hardware_optimizer_fn.reward_fns={reward_fn}",
            ],
        )
    rc = cfg.eval_fn.runner
    rc.env.robot_generator.choices = list(choices)
    rc.env.robot_generator.num_uniforms = 13
    rc.ckpt_path = expert_ckpt
    rc.env.pickle_path = cfg.pickle_path
    runner = hydra.utils.instantiate(rc, log_dir="/tmp/cmp_tt_rl")
    return runner


def rollout(runner, hardware_seed, traj_idx, max_steps):
    """한 에피소드를 굴리며 매 step 의 qpos 를 모은다."""
    env: TrackEnv = runner.env
    # 순서가 중요하다: 하드웨어를 먼저 만들어야 env.m 이 생긴다 (runner.run_episodes 와 동일)
    env.reset_hardware(hardware_seed=hardware_seed)
    runner.post_hardware_reset(hardware_seed=hardware_seed)
    env.traj_idx = traj_idx
    env.reset_no_obs(seed=traj_idx)
    env.p.forward()
    obs = env.get_obs()

    qpos_log = [env.d.qpos.copy()]
    for _ in range(max_steps):
        action = runner.default_policy(obs)
        ctrl = np.asarray(action["ctrl/target_qpos"]).reshape(-1)
        obs, _, done, _ = env.step(ctrl[: env.d.ctrl.shape[0]])
        qpos_log.append(env.d.qpos.copy())
        if done:
            break
    return env, np.array(qpos_log)


def tint(model, rgb):
    """로봇 지오메트리를 한 색으로 물들인다 (바닥·사이트는 제외)."""
    for g in range(model.ngeom):
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        a = float(model.geom_rgba[g][3])
        base = float(np.mean(model.geom_rgba[g][:3]))
        shade = 0.55 + 0.45 * base          # 원래 밝기 차이는 남긴다
        model.geom_rgba[g][:3] = np.clip(np.array(rgb) * shade, 0, 1)
        model.geom_rgba[g][3] = a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", type=int, default=9)
    p.add_argument("--steps", type=int, default=380)
    p.add_argument("--every", type=int, default=2)
    p.add_argument("--reward-fn", type=str, default="tracking_only",
                   help="TT comparison objective; default is the TT tracking reward. "
                        "Set to a different reward to isolate the effect of the RL objective.")
    p.add_argument("--designs", default=os.path.join(OUT, "compare_designs.json"))
    p.add_argument("--out-name", default="tt_vs_rl_sim")
    p.add_argument("--pkl", default="data/july25th2025-huy-20skills-test.pkl",
                   help="data/flat_walk_long.pkl 로 바꾸면 직선 전진 보행 task")
    args = p.parse_args()

    spec = json.load(open(args.designs))
    runs, models, datas, logs = {}, {}, {}, {}
    for tag in ("TT", "RL"):
        s = spec[tag]
        print(f"[{tag}] choices {s['choices']} seed {s['seed']} expert {s['expert'][:60]}...",
              flush=True)
        runner = build_runner(s["choices"], s["expert"], args.traj, args.pkl, reward_fn=args.reward_fn)
        env, q = rollout(runner, s["seed"], args.traj, args.steps)
        print(f"[{tag}] {len(q)} step, base x {q[0][0]:.3f} -> {q[-1][0]:.3f}", flush=True)
        runs[tag], models[tag], datas[tag] = runner, env.m, env.d
        logs[tag] = q

    n = min(len(logs["TT"]), len(logs["RL"]))
    # 렌더용 모델 설정: 색, 프레임버퍼, 조명
    rends, opts, cams = {}, {}, {}
    for tag in ("TT", "RL"):
        m = models[tag]
        tint(m, COLORS[tag])
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H)
        m.vis.headlight.ambient[:] = 0.5
        m.vis.headlight.diffuse[:] = 0.65
        if tag == "RL":                      # 바닥은 한 번만 그린다
            for g in range(m.ngeom):
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                    m.geom_rgba[g][3] = 0.0
        r = mujoco.Renderer(m, H, W)
        r.enable_depth_rendering()
        opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(opt); opt.geomgroup[:] = 1
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
        rends[tag], opts[tag], cams[tag] = r, opt, cam

    span = max(float(np.abs(logs[t][:, 0] - logs[t][0, 0]).max()) for t in ("TT", "RL"))
    CAM_AHEAD = 0.35 + span * 0.5
    CAM_DIST = max(2.9, 2.2 + span * 1.3)     # 많이 걸으면 카메라를 물린다
    start_x = float(logs["TT"][0][0])
    try:
        F = ImageFont.truetype(FONT, 26); Fs = ImageFont.truetype(FONT, 20)
    except OSError:
        F = Fs = ImageFont.load_default()

    frames = []
    for i in range(0, n, args.every):
        rgbs, deps = {}, {}
        for tag in ("TT", "RL"):
            d = datas[tag]
            d.qpos[:] = logs[tag][i]
            d.qpos[1] += Y_OFFSET[tag]        # 좌우로 벌려 나란히 세운다
            mujoco.mj_forward(models[tag], d)
            cam = cams[tag]
            cam.lookat[:] = [start_x + CAM_AHEAD, 0.0, 0.35]
            cam.distance, cam.azimuth, cam.elevation = CAM_DIST, 128.0, -14.0
            rends[tag].update_scene(d, cam, scene_option=opts[tag])
            rends[tag].disable_depth_rendering()
            rgbs[tag] = rends[tag].render().copy()
            rends[tag].enable_depth_rendering()
            rends[tag].update_scene(d, cam, scene_option=opts[tag])
            deps[tag] = rends[tag].render().copy()

        near = deps["RL"] < deps["TT"]        # 깊이로 합성 -> 가림 관계 유지
        comp = np.where(near[..., None], rgbs["RL"], rgbs["TT"])

        img = Image.fromarray(comp.astype(np.uint8))
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, W, 40], fill=(255, 255, 255))
        dr.text((12, 7), "TT 형상 + 그 expert", font=F,
                fill=tuple(int(255 * c) for c in COLORS["TT"]))
        dr.text((330, 7), "RL 형상 + 그 expert", font=F,
                fill=tuple(int(255 * c) for c in COLORS["RL"]))
        dx_tt = logs["TT"][i][0] - logs["TT"][0][0]
        dx_rl = logs["RL"][i][0] - logs["RL"][0][0]
        dr.text((W - 470, 10),
                f"step {i:3d}/{n}   전진 TT {dx_tt:+.3f} m · RL {dx_rl:+.3f} m",
                font=Fs, fill=(40, 40, 40))
        frames.append(np.asarray(img))

    for _ in range(25):
        frames.append(frames[-1])
    os.makedirs(OUT, exist_ok=True)
    mp4 = f"{OUT}/{args.out_name}.mp4"
    imageio.mimsave(mp4, frames, fps=25, quality=8, macro_block_size=1)
    print("saved", mp4, len(frames), "frames")


if __name__ == "__main__":
    main()
