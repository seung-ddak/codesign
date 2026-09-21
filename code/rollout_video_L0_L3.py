"""L0~L3 best design 4 개를 같은 정책·같은 시드·같은 고정 카메라로 굴려 영상 비교.

네 패널이 **완전히 같은 월드 카메라**를 쓰므로 가로 픽셀 위치 = 월드 x 가 되고,
같은 출발선에서 누가 얼마나 앞서 나갔는지 그대로 보인다 (카메라 follow 없음).

  .venv/bin/python tt_codesign/code/rollout_video_L0_L3.py
"""

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

TT = "/mnt/gstore/home/xiangyue/RL/transformer-transformer"
OUT = "/mnt/gstore/home/xiangyue/RL/tt_codesign/results"
os.chdir(TT)

from t2.eval.batched_ctrl import BatchedCtrlRollout
from t2.eval.ctrl import env_obs_to_batch

CKPT = "checkpoints/umi_on_legs_plus_plus/d7lra7j0/040.pt"
TRAJ, SEED, STEPS, GOAL = 1, 10000000, 600, 0.8
EVERY = 2                       # 2 step 마다 1 프레임
PW, PH = 1000, 300              # 패널 해상도 (가로로 길게 = 전진 거리를 보기 위함)
LBL = {"L0": "L0  prior 무작위", "L1": "L1  CMA-ES",
       "L2": "L2  prior 재학습 (TT 추종 계열)", "L3": "L3  residual guidance (우리 RL)",
       "center": "center  설계공간 중앙", "prior_2": "prior_2  정상 보행 기준선"}
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def resolve(labels):
    """라벨 -> 13 차원 설계. prior_N 은 posture_audit.py 와 동일한 난수열을 재현한다."""
    rows = {json.loads(l)["stage"]: json.loads(l)
            for l in open(f"{TT}/design_search_walk.jsonl")}
    rng = np.random.default_rng(SEED)
    pool = {"center": np.full(13, 0.5)}
    for k in range(64):
        pool[f"prior_{k}"] = rng.uniform(0, 1, 13)
    for st in ("L0", "L1", "L2", "L3"):
        pool[st] = np.array(rows[st]["best_design"], dtype=float)
    return np.array([pool[l] for l in labels], dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", default="L0,L1,L2,L3",
                   help="쉼표 구분. L0~L3 / center / prior_N")
    p.add_argument("--out-name", default="L0_L3_rollout")
    args = p.parse_args()
    stages = [x.strip() for x in args.designs.split(",")]
    designs = resolve(stages)

    t0 = time.time()
    R = BatchedCtrlRollout(ckpt=CKPT, env_name="umi_on_legs_plus_plus_from_choice",
                           pkl="data/flat_walk_long.pkl", n_envs=len(stages),
                           inference_steps=2, episode_steps=STEPS)
    print(f"평가기 준비 {time.time() - t0:.1f}s", flush=True)

    envs = R.envs[:len(stages)]
    obs_list, hw_list = [], []
    for env, design in zip(envs, designs):
        env.robot_generator = R._design_generator(design)
        env.reset_hardware(SEED)
        obs, _, hw = env.reset_sim_and_target_traj(seed=SEED, target_traj_idx=TRAJ)
        obs_list.append(obs)
        hw_list.append(hw)

    # 렌더러 + 전 패널 공통 고정 카메라
    rends, opts, cams = [], [], []
    start_x = np.array([float(e.d.qpos[0]) for e in envs])
    for env in envs:
        # 모델 XML 에 offwidth 가 없으므로 컴파일 후 프레임버퍼를 직접 키운다
        env.m.vis.global_.offwidth = max(env.m.vis.global_.offwidth, PW)
        env.m.vis.global_.offheight = max(env.m.vis.global_.offheight, PH)
        env.m.vis.headlight.ambient[:] = 0.55      # 기본 씬이 너무 어둡다
        env.m.vis.headlight.diffuse[:] = 0.65
        env.m.vis.headlight.specular[:] = 0.1
        r = mujoco.Renderer(env.m, PH, PW)
        opt = mujoco.MjvOption(); mujoco.mjv_defaultOption(opt)
        opt.geomgroup[:] = 1                    # 링크 캡슐이 group 3
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = [start_x[0] + 0.85, 0.0, 0.28]
        cam.distance, cam.azimuth, cam.elevation = 2.6, 90.0, -7.0
        rends.append(r); opts.append(opt); cams.append(cam)

    try:
        F = ImageFont.truetype(FONT, 22)
        Fs = ImageFont.truetype(FONT, 18)
    except OSError:
        F = Fs = ImageFont.load_default()

    prev_actions = [np.zeros(e.d.ctrl.shape[0]) for e in envs]
    alive = np.ones(len(envs), dtype=bool)
    max_x = start_x.copy()
    fell_at = [None] * len(envs)
    frames = []

    # ---- 자세 진단: "걸어서" 간 것인지 "깔려서 미끄러진" 것인지 구분한다 -------------
    base_bid = [mujoco.mj_name2id(e.m, mujoco.mjtObj.mjOBJ_BODY, "base") for e in envs]
    diag = [[] for _ in envs]

    def diag_step(i):
        env = envs[i]
        m, d = env.m, env.d
        body_ground = 0          # 발이 아닌 링크가 바닥에 닿은 접촉 수
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            t1, t2 = m.geom_type[g1], m.geom_type[g2]
            if t1 == mujoco.mjtGeom.mjGEOM_PLANE:
                g = g2
            elif t2 == mujoco.mjtGeom.mjGEOM_PLANE:
                g = g1
            else:
                continue
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""
            if "foot" not in bn:
                body_ground += 1
        diag[i].append((float(d.qpos[2]),                       # 베이스 높이
                        float(d.xmat[base_bid[i]].reshape(3, 3)[2, 2]),   # 기립도 (1 = 수평)
                        body_ground,
                        float(d.qpos[0] - start_x[i])))

    def grab(step):
        panels = []
        for i, env in enumerate(envs):
            rends[i].update_scene(env.d, cams[i], scene_option=opts[i])
            sc = rends[i].scene
            for gx, rgba in ((start_x[i], (0.55, 0.55, 0.60, 1.0)),
                             (start_x[i] + GOAL, (0.15, 0.80, 0.35, 1.0))):
                if sc.ngeom < sc.maxgeom:
                    mujoco.mjv_initGeom(
                        sc.geoms[sc.ngeom], mujoco.mjtGeom.mjGEOM_BOX,
                        np.array([0.006, 0.6, 0.002]),
                        np.array([gx, 0.0, 0.002]), np.eye(3).flatten(),
                        np.array(rgba, dtype=np.float32))
                    sc.ngeom += 1
            img = Image.fromarray(rends[i].render())
            dr = ImageDraw.Draw(img)
            dist = max_x[i] - start_x[i]
            dr.rectangle([0, 0, PW, 30], fill=(255, 255, 255))
            dr.text((10, 4), LBL.get(stages[i], stages[i]), font=F, fill=(15, 15, 15))
            state = "낙상" if fell_at[i] is not None else ("도달" if dist >= GOAL else "주행")
            col = (200, 40, 40) if state == "낙상" else (
                (25, 130, 60) if state == "도달" else (60, 60, 60))
            dr.text((PW - 300, 6), f"전진 {dist:0.3f} m   {state}", font=Fs, fill=col)
            dr.line([(0, 31), (PW, 31)], fill=(210, 210, 210), width=1)
            panels.append(np.asarray(img))
        f = np.concatenate(panels, axis=0)
        bar = Image.new("RGB", (f.shape[1], 34), "white")
        d2 = ImageDraw.Draw(bar)
        d2.text((10, 6), f"step {step:3d}/{STEPS}   t = {step * envs[0].ctrl_dt:5.2f} s"
                         f"   목표 {GOAL} m   명령 0.10 m/s   고정 카메라 (follow 없음)",
                font=Fs, fill=(40, 40, 40))
        frames.append(np.concatenate([np.asarray(bar), f], axis=0))

    grab(0)
    t0 = time.time()
    for step in range(STEPS):
        idx = np.flatnonzero(alive)
        if len(idx) == 0:
            break
        batch = env_obs_to_batch(
            obs_list=[{**R._reshape_obs(envs[i], obs_list[i]), **hw_list[i]} for i in idx],
            prev_action_list=[prev_actions[i] for i in idx],
            seq_len_cfg=R.seq_len_cfg, device=R.device,
            group_time_offsets=R.offsets, batch_process_fn=R.batch_process_fn)
        with torch.inference_mode():
            out = R.decoder(batch)
        actions = (out["ctrl/target_qpos"].float().cpu().numpy()
                   .reshape(len(idx), R.rollout_steps, R.seq_len_cfg["actuator"]))
        for j, i in enumerate(idx):
            env = envs[i]
            a = actions[j, R.num_past][: env.d.ctrl.shape[0]]
            prev_actions[i] = a
            obs_list[i], _, done, _ = env.step(a)
            max_x[i] = max(max_x[i], float(env.d.qpos[0]))
            diag_step(i)
            if done:
                alive[i] = False
                fell_at[i] = step + 1
        if (step + 1) % EVERY == 0:
            grab(step + 1)
        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{STEPS}  {time.time() - t0:.0f}s  "
                  f"dist={np.round(max_x - start_x, 3).tolist()}", flush=True)

    # 마지막 프레임을 1 초간 유지
    for _ in range(25):
        frames.append(frames[-1])

    os.makedirs(OUT, exist_ok=True)
    mp4 = f"{OUT}/{args.out_name}.mp4"
    imageio.mimsave(mp4, frames, fps=25, quality=8, macro_block_size=1)

    res = {}
    for i, st in enumerate(stages):
        a = np.array(diag[i]) if diag[i] else np.zeros((1, 4))
        half = a[len(a) // 2:]                      # 후반부 = 정상 상태
        res[st] = dict(
            distance=round(float(max_x[i] - start_x[i]), 4),
            success=bool(max_x[i] - start_x[i] >= GOAL),
            fell_at_step=fell_at[i],
            steps=STEPS if fell_at[i] is None else fell_at[i],
            base_z_mean_2nd_half=round(float(half[:, 0].mean()), 4),
            base_z_min=round(float(a[:, 0].min()), 4),
            upright_mean_2nd_half=round(float(half[:, 1].mean()), 4),
            body_ground_contact_frac=round(float((half[:, 2] > 0).mean()), 4),
            body_ground_contact_mean=round(float(half[:, 2].mean()), 2),
        )
    print(json.dumps(res, indent=2, ensure_ascii=False))
    with open(f"{OUT}/{args.out_name}.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("saved", mp4, len(frames), "frames")


if __name__ == "__main__":
    main()
